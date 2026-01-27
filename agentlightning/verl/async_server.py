# Copyright (c) Microsoft. All rights reserved.

# type: ignore

from copy import deepcopy

import ray
from verl.workers.rollout.sglang_rollout.async_sglang_server import SGLangHttpServer
from verl.workers.rollout.vllm_rollout.vllm_async_server import vLLMHttpServer

from agentlightning.instrumentation.sglang import instrument_sglang
from agentlightning.instrumentation.vllm import instrument_vllm


def _unwrap_ray_remote(cls):
    if hasattr(cls, "__ray_actor_class__"):
        cls = cls.__ray_actor_class__
    return cls


class PatchedvLLMHttpServer(vLLMHttpServer):

    def __init__(self, *args, **kwargs):
        instrument_vllm()
        super().__init__(*args, **kwargs)

        self.config = deepcopy(self.config)
        self.config.rollout.multi_turn.tool_config_path = "/dev/null"


@ray.remote(num_cpus=1)
class PatchedSGLangHttpServer(_unwrap_ray_remote(SGLangHttpServer)):

    def __init__(self, *args, **kwargs):
        instrument_sglang()
        super().__init__(*args, **kwargs)

        self.config = deepcopy(self.config)
        self.config.rollout.multi_turn.tool_config_path = "/dev/null"
