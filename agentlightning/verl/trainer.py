# Copyright (c) Microsoft. All rights reserved.

# type: ignore

from __future__ import annotations

import gc
import random
from copy import deepcopy
from pprint import pprint
from typing import Any, Dict, Type

import numpy as np
import torch
import verl
from tqdm import tqdm
from verl import DataProto
from verl.experimental.dataset.sampler import AbstractCurriculumSampler
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.trainer.ppo.core_algos import agg_loss
from verl.trainer.ppo.metric_utils import _compute_response_info, compute_throughout_metrics, compute_timing_metrics
from verl.trainer.ppo.ray_trainer import (
    AdvantageEstimator,
    RayPPOTrainer,
    Role,
    apply_kl_penalty,
    compute_advantage,
    compute_response_mask,
)
from verl.utils.checkpoint.checkpoint_manager import should_save_ckpt_esi
from verl.utils.debug import marked_timer
from verl.utils.metric import reduce_metrics
from verl.utils.rollout_skip import RolloutSkip

from agentlightning.adapter import TraceAdapter, TraceToTripletBase
from agentlightning.llm_proxy import LLMProxy
from agentlightning.store.base import LightningStore

from .daemon import AgentModeDaemon

__all__ = [
    "AgentLightningTrainer",
]


# This function is adapted from verl.
# We introduce a new parameter `suffix` to distinguish between metrics computed
# before and after AgentLightning’s post-processing.
# - "Before" refers to raw reward and advantage values.
# - "After" refers to values computed following post-processing, which involves:
#     (1) Dropping prompts that exceed the maximum allowed length.
#     (2) Adjusting the batch size to be a multiple of the mini PPO size.
# Different suffixes are used to label these two stages accordingly.
def compute_data_metrics(batch: DataProto, use_critic: bool = True, suffix: str = "") -> Dict[str, Any]:
    """
    Computes various metrics from a batch of data for PPO training.

    This function calculates metrics related to scores, rewards, advantages, returns, values,
    and sequence lengths from a batch of data. It provides statistical information (mean, max, min)
    for each metric category.

    Args:
        batch: A DataProto object containing batch data with token-level scores, rewards, advantages, etc.
        use_critic: Whether to include critic-specific metrics. Defaults to True.

    Returns:
        A dictionary of metrics including:
            - critic/score/mean, max, min: Statistics about sequence scores
            - critic/rewards/mean, max, min: Statistics about sequence rewards
            - critic/advantages/mean, max, min: Statistics about advantages
            - critic/returns/mean, max, min: Statistics about returns
            - critic/values/mean, max, min: Statistics about critic values (if use_critic=True)
            - critic/vf_explained_var: Explained variance of the value function (if use_critic=True)
            - response_length/mean, max, min, clip_ratio: Statistics about response lengths
            - prompt_length/mean, max, min, clip_ratio: Statistics about prompt lengths
    """
    sequence_score = batch.batch["token_level_scores"].sum(-1)
    sequence_reward = batch.batch["token_level_rewards"].sum(-1)

    advantages = batch.batch["advantages"]
    returns = batch.batch["returns"]

    max_response_length = batch.batch["responses"].shape[-1]

    prompt_mask = batch.batch["attention_mask"][:, :-max_response_length].bool()
    response_mask = batch.batch["attention_mask"][:, -max_response_length:].bool()

    max_prompt_length = prompt_mask.size(-1)

    response_info = _compute_response_info(batch)
    prompt_length = response_info["prompt_length"]
    response_length = response_info["response_length"]

    valid_adv = torch.masked_select(advantages, response_mask)
    valid_returns = torch.masked_select(returns, response_mask)

    return_var = torch.tensor(0.0)
    return_diff_var = torch.tensor(0.0)
    if use_critic:
        values = batch.batch["values"]
        valid_values = torch.masked_select(values, response_mask)
        return_diff_var = torch.var(valid_returns - valid_values)
        return_var = torch.var(valid_returns)

    metrics = {
        # score
        "critic/score/mean" + suffix: torch.mean(sequence_score).detach().item(),
        "critic/score/max" + suffix: torch.max(sequence_score).detach().item(),
        "critic/score/min" + suffix: torch.min(sequence_score).detach().item(),
        # reward
        "critic/rewards/mean" + suffix: torch.mean(sequence_reward).detach().item(),
        "critic/rewards/max" + suffix: torch.max(sequence_reward).detach().item(),
        "critic/rewards/min" + suffix: torch.min(sequence_reward).detach().item(),
        # adv
        "critic/advantages/mean" + suffix: torch.mean(valid_adv).detach().item(),
        "critic/advantages/max" + suffix: torch.max(valid_adv).detach().item(),
        "critic/advantages/min" + suffix: torch.min(valid_adv).detach().item(),
        # returns
        "critic/returns/mean" + suffix: torch.mean(valid_returns).detach().item(),
        "critic/returns/max" + suffix: torch.max(valid_returns).detach().item(),
        "critic/returns/min" + suffix: torch.min(valid_returns).detach().item(),
        **(
            {
                # values
                "critic/values/mean" + suffix: torch.mean(valid_values).detach().item(),
                "critic/values/max" + suffix: torch.max(valid_values).detach().item(),
                "critic/values/min" + suffix: torch.min(valid_values).detach().item(),
                # vf explained var
                "critic/vf_explained_var" + suffix: (1.0 - return_diff_var / (return_var + 1e-5)).detach().item(),
            }
            if use_critic
            else {}
        ),
        # response length
        "response_length/mean" + suffix: torch.mean(response_length).detach().item(),
        "response_length/max" + suffix: torch.max(response_length).detach().item(),
        "response_length/min" + suffix: torch.min(response_length).detach().item(),
        "response_length/clip_ratio" + suffix: torch.mean(torch.eq(response_length, max_response_length).float()).detach().item(),
        # prompt length
        "prompt_length/mean" + suffix: torch.mean(prompt_length).detach().item(),
        "prompt_length/max" + suffix: torch.max(prompt_length).detach().item(),
        "prompt_length/min" + suffix: torch.min(prompt_length).detach().item(),
        "prompt_length/clip_ratio" + suffix: torch.mean(torch.eq(prompt_length, max_prompt_length).float()).detach().item(),
    }
    return metrics


class AgentLightningTrainer(RayPPOTrainer):
    """
    Specialized PPO trainer for agent-based reinforcement learning.

    This trainer is designed specifically for scenarios where the model interacts with
    external environments, tools, or APIs through an AgentLightningServer. It simplifies
    the training loop by removing the complex conditional logic present in the original
    RayPPOTrainer and focusing on the agent mode workflow.

    Key differences from RayPPOTrainer:

    1. Uses AgentModeDaemon for server communication
    2. Simplified data flow without pop/union operations
    3. Direct batch processing through agent daemon
    4. Streamlined validation using agent_mode validation
    """

    def __init__(
        self,
        store: LightningStore | None,
        llm_proxy: LLMProxy | None,
        llm_proxy_port: int | None,
        adapter: TraceAdapter | None,
        daemon_cls: Type[AgentModeDaemon],
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.store = store
        self.llm_proxy = llm_proxy
        self.llm_proxy_port = llm_proxy_port
        self.adapter = adapter
        self.daemon_cls = daemon_cls

    def _validate(self):
        assert len(self.val_dataloader) == 1, "Please set val_batch_size to None for better throughput."

        test_data = next(iter(self.val_dataloader))
        test_batch = DataProto.from_single_dict(test_data)
        self.agent_mode_daemon.set_up_data_and_server(
            test_batch.non_tensor_batch,
            self.async_rollout_manager.server_addresses,
            is_train=False,
        )
        self.agent_mode_daemon.run_until_all_finished()
        test_metrics = self.agent_mode_daemon.get_test_metrics()
        self.agent_mode_daemon.clear_data_and_server()
        return test_metrics

    def _train_step(self, batch_dict: dict, curr_step_profile: bool = False, is_last_step: bool = False) -> dict:
        # Isolate in a separate method to automatically recycle the variables before validation.

        batch: DataProto = DataProto.from_single_dict(batch_dict)
        metrics = {}
        timing_raw = {}

        with marked_timer("step", timing_raw):
            # When agent mode is enabled, we read the batch as it is.
            gen_batch = batch

            # generate a batch
            with marked_timer("gen", timing_raw, color="red"):
                if curr_step_profile:
                    self.async_rollout_manager.start_profile()
                self.agent_mode_daemon.set_up_data_and_server(
                    gen_batch.non_tensor_batch, self.async_rollout_manager.server_addresses
                )
                self.agent_mode_daemon.run_until_all_finished()
                batch, agent_metrics = self.agent_mode_daemon.get_train_data_batch(
                    max_prompt_length=(
                        self.config.agentlightning.trace_aggregator.trajectory_max_prompt_length
                        if self.config.agentlightning.trace_aggregator.level.startswith("trajectory")
                        else self.config.data.max_prompt_length
                    ),
                    max_response_length=(
                        self.config.agentlightning.trace_aggregator.trajectory_max_response_length
                        if self.config.agentlightning.trace_aggregator.level.startswith("trajectory")
                        else self.config.data.max_response_length
                    ),
                    device=gen_batch.batch["fake_ids"].device,
                    global_steps=self.global_steps,
                )
                metrics.update(agent_metrics)
                self.agent_mode_daemon.clear_data_and_server()
                self.checkpoint_manager.sleep_replicas()
                if curr_step_profile:
                    self.async_rollout_manager.stop_profile()

            if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
                with marked_timer("gen_max", timing_raw, color="purple"):
                    gen_baseline_batch = deepcopy(gen_batch)
                    gen_baseline_batch.meta_info["do_sample"] = False
                    if curr_step_profile:
                        self.async_rollout_manager.start_profile()
                    gen_baseline_output = self.async_rollout_manager.generate_sequences(gen_baseline_batch)
                    self.checkpoint_manager.sleep_replicas()
                    if curr_step_profile:
                        self.async_rollout_manager.stop_profile()

                    batch = batch.union(gen_baseline_output)

                    if self.use_rm and "rm_scores" not in batch.batch.keys():
                        batch_reward = self._compute_reward_colocate(batch)
                        batch = batch.union(batch_reward)

                        # Compute or extract reward for REMAX baseline
                        reward_baseline_tensor = batch.batch["rm_scores"].sum(dim=-1)

                        keys_to_pop = set(gen_baseline_output.batch.keys())
                        if batch_reward is not None:
                            keys_to_pop.update(batch_reward.batch.keys())
                        batch.pop(batch_keys=list(keys_to_pop))

                        batch.batch["reward_baselines"] = reward_baseline_tensor
                        del batch_reward

                    del gen_baseline_batch, gen_baseline_output

            # Release the original input batch to free memory now that
            # training data has been extracted from the daemon.
            del gen_batch

            # uid is used for algorithm like GRPO, should be aligned to data id
            batch.non_tensor_batch["uid"] = batch.non_tensor_batch["data_id_list"]

            if "response_mask" not in batch.batch.keys():
                batch.batch["response_mask"] = compute_response_mask(batch)

            # compute global_valid tokens
            batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

            with marked_timer("reward", timing_raw, color="yellow"):
                # compute reward model score
                if self.use_rm and "rm_scores" not in batch.batch.keys():
                    from verl.trainer.ppo.reward import extract_reward

                    batch_reward = self._compute_reward_colocate(batch)
                    batch = batch.union(batch_reward)

                    reward_tensor, reward_extra_infos_dict = extract_reward(batch)

                    if hasattr(self, "_maybe_build_self_distillation_batch"):
                        self_distillation_data = self._maybe_build_self_distillation_batch(
                            batch,
                            reward_tensor,
                            reward_extra_infos_dict,
                        )
                        if self_distillation_data is not None:
                            self_distillation_batch, self_distillation_metrics = self_distillation_data
                            batch = batch.union(self_distillation_batch)
                            metrics.update(self_distillation_metrics)

            # for agent mode, pad the lengths to calculate old log prob, ref, and values
            batch, pad_size = pad_dataproto_to_divisor(batch, self.actor_rollout_wg.world_size)

            # Operating Mode Selection:
            # - Bypass mode: Sets old_log_probs = rollout_log_probs (2 policies: π_rollout, π_θ)
            # - Decoupled mode: Recomputes old_log_probs as proximal anchor (3 policies: π_rollout, π_old, π_θ)
            #   Note: π_old computed once per data batch, serves as stable reference during mini-batch updates
            rollout_corr_config = self.config.algorithm.get("rollout_correction", None)
            bypass_recomputing_logprobs = rollout_corr_config and rollout_corr_config.get("bypass_mode", False)
            if bypass_recomputing_logprobs:  # Use `rollout_log_probs`
                from verl.trainer.ppo.rollout_corr_helper import apply_bypass_mode

                apply_bypass_mode(
                    batch=batch,
                    rollout_corr_config=rollout_corr_config,
                    policy_loss_config=self.config.actor_rollout_ref.actor.policy_loss,
                )
            else:
                # recompute old_log_probs
                with marked_timer("old_log_prob", timing_raw, color="blue"):
                    # old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                    old_log_prob, old_log_prob_mfu = self._compute_old_log_prob(batch)
                    entropys = old_log_prob.batch["entropys"]
                    response_masks = batch.batch["response_mask"]
                    actor_config = self.config.actor_rollout_ref.actor
                    entropy_loss = agg_loss(
                        loss_mat=entropys, loss_mask=response_masks, loss_agg_mode=actor_config.loss_agg_mode,
                        loss_scale_factor=actor_config.loss_scale_factor
                    )
                    old_log_prob_metrics = {
                        "actor/entropy_loss": entropy_loss.detach().item(),
                        "perf/mfu/actor_infer": old_log_prob_mfu,
                    }
                    # old_log_prob_metrics = {"actor/entropy_loss": entropy_loss.detach().item()}
                    metrics.update(old_log_prob_metrics)
                    old_log_prob.batch.pop("entropys")
                    if "routed_experts" in batch.batch and "routed_experts" in old_log_prob.batch:
                        raise ValueError(
                            "Detected conflicting router replay configuration: "
                            "router_replay.mode='R2' and enable_rollout_routing_replay=True "
                            "cannot be enabled simultaneously. "
                            "The enable_rollout_routing_replay option is only used in R3 mode; "
                            "it should not be set when using R2 mode."
                        )
                    batch = batch.union(old_log_prob)
                    if "rollout_log_probs" in batch.batch.keys():
                        # TODO: we may want to add diff of probs too.
                        from verl.utils.debug.metrics import calculate_debug_metrics

                        metrics.update(calculate_debug_metrics(batch))

            assert "old_log_probs" in batch.batch, f'"old_log_prob" not in {batch.batch.keys()=}'

            if self.use_reference_policy:
                # compute reference log_prob
                with marked_timer(str(Role.RefPolicy), timing_raw, color="olive"):
                    ref_log_prob = self._compute_ref_log_prob(batch)
                    batch = batch.union(ref_log_prob)

            # compute values
            if self.use_critic:
                with marked_timer("values", timing_raw, color="cyan"):
                    values = self._compute_values(batch)
                    batch = batch.union(values)

            # for agent mode, unpad to calculate adv
            # it is important, as adv should be based on the raw traces
            batch = unpad_dataproto(batch, pad_size=pad_size)

            with marked_timer("adv", timing_raw, color="brown"):
                # if agent_mode is enabled, there is already token_level_scores
                # token_level_scores is not needed to compute here

                reward_extra_infos_dict: dict[str, list]
                batch.batch["token_level_scores"] = reward_tensor

                if reward_extra_infos_dict:
                    batch.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})

                # compute rewards. apply_kl_penalty if available
                if self.config.algorithm.use_kl_in_reward:
                    batch, kl_metrics = apply_kl_penalty(
                        batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty
                    )
                    metrics.update(kl_metrics)
                else:
                    batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

                # Compute rollout correction: IS weights, rejection sampling, and metrics
                # Only runs in decoupled mode (computes once per batch using stable π_old)
                # In bypass mode, this is skipped - actor computes metrics from evolving π_θ vs π_rollout
                if (
                    rollout_corr_config is not None and "rollout_log_probs" in batch.batch and not bypass_recomputing_logprobs  # Only in decoupled mode
                ):
                    from verl.trainer.ppo.rollout_corr_helper import compute_rollout_correction_and_add_to_batch

                    # Compute IS weights, apply rejection sampling, compute metrics
                    batch, is_metrics = compute_rollout_correction_and_add_to_batch(batch, rollout_corr_config)
                    # IS and off-policy metrics already have rollout_corr/ prefix
                    metrics.update(is_metrics)

                # compute advantages, executed on the driver process

                norm_adv_by_std_in_grpo = self.config.algorithm.get(
                    "norm_adv_by_std_in_grpo", True
                )  # GRPO adv normalization factor

                batch = compute_advantage(
                    batch,
                    adv_estimator=self.config.algorithm.adv_estimator,
                    gamma=self.config.algorithm.gamma,
                    lam=self.config.algorithm.lam,
                    num_repeat=self.config.actor_rollout_ref.rollout.n,
                    norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                    config=self.config.algorithm,
                )

            # Calculate the metrics before processing. Refer to the comments of function `compute_data_metrics` for details.
            metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic, suffix="_before_processing"))

            # after advantages are assigned, we begin to drop (1) long prompt (2) floor to ppo minisize
            keep_indices = (~batch.batch["is_drop_mask"]).nonzero(as_tuple=True)[0]
            metrics["training/n_triplets_prompt_too_long"] = (
                batch.batch["is_drop_mask"].shape[0] - keep_indices.shape[0]
            )
            batch = batch[keep_indices]
            # next, round to minibatch size
            mini_batch_size = self.config.actor_rollout_ref.actor.ppo_mini_batch_size
            n_transition = len(batch)
            random_indices = list(range(n_transition))
            random.shuffle(random_indices)
            batch.reorder(torch.tensor(random_indices).type(torch.int32))
            n_remained_transition = n_transition // mini_batch_size * mini_batch_size
            batch = batch[list(range(n_remained_transition))]
            metrics["training/n_triplets_dropped_remainder"] = n_transition - n_remained_transition

            # Agent mode note: Change the order of balance batch;
            #     1. first calculate advantage
            #     2. then drop the samples (too long prompt & floor to ppo minisize)
            #     3. balance
            # balance the number of valid tokens on each dp rank.
            # Note that this breaks the order of data inside the batch.
            # Please take care when you implement group based adv computation such as GRPO and rloo
            if self.config.trainer.balance_batch:
                self._balance_batch(batch, metrics=metrics)

            # update critic
            if self.use_critic:
                with marked_timer("update_critic", timing_raw, color="pink"):
                    critic_output = self._update_critic(batch)
                critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                metrics.update(critic_output_metrics)

            # implement critic warmup
            if self.config.trainer.critic_warmup <= self.global_steps:
                # update actor
                with marked_timer("update_actor", timing_raw, color="red"):
                    batch.meta_info["multi_turn"] = self.config.actor_rollout_ref.rollout.multi_turn.enable
                    actor_output = self._update_actor(batch)

                # Check if the ESI (Elastic Server Instance)/training plan is close to expiration.
                esi_close_to_expiration = should_save_ckpt_esi(
                    max_steps_duration=self.max_steps_duration,
                    redundant_time=self.config.trainer.esi_redundant_time,
                )

                # Check if the conditions for saving a checkpoint are met.
                # The conditions include a mandatory condition (1) and
                # one of the following optional conditions (2/3/4):
                # 1. The save frequency is set to a positive value.
                # 2. It's the last training step.
                # 3. The current step number is a multiple of the save frequency.
                # 4. The ESI(Elastic Server Instance)/training plan is close to expiration.
                if self.config.trainer.save_freq > 0 and (
                    is_last_step or self.global_steps % self.config.trainer.save_freq == 0 or esi_close_to_expiration
                ):
                    if esi_close_to_expiration:
                        print("Force saving checkpoint: ESI instance expiration approaching.")
                    with marked_timer("save_checkpoint", timing_raw, color="green"):
                        self._save_checkpoint()

                # update weights from trainer to rollout
                with marked_timer("update_weights", timing_raw, color="red"):
                    self.checkpoint_manager.update_weights(self.global_steps)

                actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                metrics.update(actor_output_metrics)

            # Log rollout generations if enabled
            rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
            if rollout_data_dir:
                self._log_rollout_data(batch, reward_extra_infos_dict, timing_raw, rollout_data_dir)

        # compute training metrics
        metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic, suffix="_after_processing"))
        metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
        # TODO: implement actual tflpo and theoretical tflpo
        n_gpus = self.resource_pool_manager.get_n_gpus()
        metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))

        # GDPO per-component reward metrics
        gdpo_reward_keys = self.config.algorithm.get("gdpo_reward_keys", None)
        if gdpo_reward_keys and self.config.algorithm.adv_estimator in ("gdpo", AdvantageEstimator.GDPO):
            for key in gdpo_reward_keys:
                if key in batch.non_tensor_batch:
                    vals = np.asarray(batch.non_tensor_batch[key], dtype=np.float32)
                    metrics[f"gdpo/{key}/mean"] = float(np.mean(vals))
                    metrics[f"gdpo/{key}/std"] = float(np.std(vals))
                    metrics[f"gdpo/{key}/max"] = float(np.max(vals))
                    metrics[f"gdpo/{key}/min"] = float(np.min(vals))

        # this is experimental and may be changed/removed in the future in favor of a general-purpose one
        if isinstance(self.train_dataloader.sampler, AbstractCurriculumSampler):
            self.train_dataloader.sampler.update(batch=batch)

        # this is experimental and may be changed/removed in the future
        # in favor of a general-purpose data buffer pool
        if hasattr(self.train_dataset, "on_batch_end"):
            # The dataset may be changed after each training batch
            self.train_dataset.on_batch_end(batch=batch)

        # Explicitly release batch tensors and trigger garbage collection to
        # prevent memory accumulation across training steps.
        del batch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return metrics

    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC
        to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        from omegaconf import OmegaConf
        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0

        # load checkpoint before doing anything
        self._load_checkpoint()
        self.checkpoint_manager.update_weights(self.global_steps)

        assert self.async_rollout_mode, "If agent mode is enabled, async server must be enabled"
        if self.adapter is not None and not isinstance(self.adapter, TraceToTripletBase):
            raise ValueError("Adapter must be a TraceToTripletBase for currently VERL implementation.")
        verl_version = verl.__version__
        if verl_version == "0.5.0":
            # Note (Zhiyuan): To avoid further patch into vllm async server, using the same sentence to get the naming here.
            # However, it is possible that verl updates the naming and causes incompatibility.
            # Reference: https://github.com/volcengine/verl/blob/5b5e09d9cc20625e436d01f69d9cc739ff681c54/verl/workers/rollout/vllm_rollout/vllm_async_server.py#L217
            model = "/".join(self.config.actor_rollout_ref.model.path.split("/")[-2:])
        else:
            # For other versions (e.g., 0.6.0), we use the full path to the model.
            model = self.config.actor_rollout_ref.model.path
        self.agent_mode_daemon = self.daemon_cls(
            self.config.agentlightning.port,
            self.config.actor_rollout_ref.rollout.n,
            train_information={
                "model": model,
                "temperature": self.config.actor_rollout_ref.rollout.temperature,
            },
            tokenizer=self.tokenizer,
            mini_batch_size=self.config.actor_rollout_ref.actor.ppo_mini_batch_size,
            pad_token_id=self.tokenizer.pad_token_id,
            mode="v1" if self.store is not None else "v0",
            store=self.store,
            llm_proxy=self.llm_proxy,
            llm_proxy_port=self.llm_proxy_port,
            adapter=self.adapter,
            processor=self.processor,  # For Qwen2-VL mrope position_ids
            image_base_dir=getattr(self.config.data, "image_base_dir", None),
            trace_aggregator=self.config.agentlightning.trace_aggregator,
        )
        self.agent_mode_daemon.start()

        current_epoch = self.global_steps // len(self.train_dataloader)

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        if self.config.actor_rollout_ref.rollout.get("skip_rollout", False):
            rollout_skip = RolloutSkip(self.config, self.async_rollout_manager)
            rollout_skip.wrap_generate_sequences()

        # add tqdm
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

        # we start from step 1
        self.global_steps += 1
        last_val_metrics = None
        self.max_steps_duration = 0

        prev_step_profile = False
        curr_step_profile = (
            self.global_steps in self.config.global_profiler.steps
            if self.config.global_profiler.steps is not None
            else False
        )
        next_step_profile = False

        for epoch in range(current_epoch, self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                if hasattr(self.actor_rollout_wg, "async_calls_finalize_fn_exec"):
                    self.actor_rollout_wg.async_calls_finalize_fn_exec(blocking=False)

                metrics = {}
                timing_raw = {}

                is_last_step = self.global_steps >= self.total_training_steps

                with marked_timer("start_profile", timing_raw):
                    self._start_profiling(
                        not prev_step_profile and curr_step_profile
                        if self.config.global_profiler.profile_continuous_steps
                        else curr_step_profile
                    )

                # train step
                metrics = self._train_step(batch_dict, curr_step_profile=curr_step_profile, is_last_step=is_last_step)

                # validate
                if self.config.trainer.test_freq > 0 and (
                    is_last_step or self.global_steps % self.config.trainer.test_freq == 0
                ):
                    with marked_timer("testing", timing_raw, color="green"):
                        val_metrics: dict = self._validate()
                        if is_last_step:
                            last_val_metrics = val_metrics
                    metrics.update(val_metrics)

                with marked_timer("stop_profile", timing_raw):
                    next_step_profile = (
                        self.global_steps + 1 in self.config.global_profiler.steps
                        if self.config.global_profiler.steps is not None
                        else False
                    )
                    self._stop_profiling(
                        curr_step_profile and not next_step_profile
                        if self.config.global_profiler.profile_continuous_steps
                        else curr_step_profile
                    )
                    prev_step_profile = curr_step_profile
                    curr_step_profile = next_step_profile

                steps_duration = timing_raw["step"]
                self.max_steps_duration = max(self.max_steps_duration, steps_duration)

                # step metrics
                metrics.update(
                    {
                        "training/global_step": self.global_steps,
                        "training/epoch": epoch,
                    }
                )

                # TODO: make a canonical logger that supports various backend
                logger.log(data=metrics, step=self.global_steps)

                progress_bar.update(1)
                self.global_steps += 1

                if (
                    hasattr(self.config.actor_rollout_ref.actor, "profiler") and self.config.actor_rollout_ref.actor.profiler.tool == "torch_memory"
                ):
                    self.actor_rollout_wg.dump_memory_snapshot(
                        tag=f"post_update_step{self.global_steps}", sub_dir=f"step{self.global_steps}"
                    )

                if is_last_step:
                    if hasattr(self.actor_rollout_wg, "async_calls_finalize_fn_exec"):
                        self.actor_rollout_wg.async_calls_finalize_fn_exec(blocking=True)
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    return
