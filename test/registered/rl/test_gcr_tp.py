"""Test GCR suspend/resume with TP=2 (NCCL collective communication between GPUs)."""

import os
import unittest

import sglang as sgl
from sglang.test.test_utils import get_gpu_count

MODEL_PATH = "/root/models/Qwen2.5-0.5B-Instruct"
DEFAULT_GCR_PRELOAD_PATH = "/root/GCR/GCR/libpreload.so:/root/GCR/GCR/libcuda.so"


def _generate_and_check(engine):
    output = engine.generate("The capital of the world is", {"max_new_tokens": 8})
    assert len(output["text"]) > 0, "Empty generation output"
    print("output:", output["text"])
    return output


class TestGcrTp(unittest.TestCase):
    """Test GCR suspend/resume with TP=2 where NCCL collectives are active."""

    def test_engine_tp2(self):
        if get_gpu_count() < 2:
            self.skipTest("Need at least 2 GPUs for tp_size=2")
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0,3")
        os.environ.setdefault("GCR_PRELOAD_PATH", DEFAULT_GCR_PRELOAD_PATH)
        os.environ.setdefault("NCCL_CUMEM_ENABLE", "1")
        # os.environ.setdefault("NCCL_SHM_DISABLE", "1")
        os.environ["NCCL_NET_DISABLE"] = "1"
        os.environ["NCCL_IB_DISABLE"] = "1"
        os.environ["NCCL_DEBUG"] = "INFO"
        os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
        os.environ["TORCH_USE_CUDA_DSA"] = "1"
        # os.environ["NCCL_DEBUG"] = "WARN"

        engine = sgl.Engine(
            model_path=MODEL_PATH,
            random_seed=42,
            enable_gcr=True,
            tp_size=2,
            # disable_custom_all_reduce=True,  # disable custom all reduce when device does not support p2p communication
        )

        _generate_and_check(engine)

        engine.gcr_suspend()
        engine.gcr_resume()

        _generate_and_check(engine)
        engine.shutdown()


if __name__ == "__main__":
    unittest.main()
