"""Test that GCR preload library is correctly attached to scheduler worker processes."""

import os
from signal import SIGTERM
import unittest

from blinker import signal

from sglang.srt.grpc.scheduler_launcher import launch_scheduler_process_only
import sglang as sgl
from sglang.test.test_utils import get_gpu_count



MODEL_PATH = "/root/models/Qwen2.5-0.5B-Instruct"
DEFAULT_GCR_PRELOAD_PATH = "/root/GCR/GCR/libpreload.so:/root/GCR/GCR/libcuda.so"

def _generate_and_check(engine):
    output = engine.generate("The capital of France is", {"max_new_tokens": 8})
    assert len(output["text"]) > 0, "Empty generation output"
    print("output:", output["text"])
    return output


class TestGcrPreloadEngine(unittest.TestCase):
    """Test GCR preload via the standard sgl.Engine path (engine.py)."""
  
    def test_engine_dp2(self):
        if get_gpu_count() < 2:
            self.skipTest("Need at least 2 GPUs for dp_size=2")
        os.environ.setdefault("GCR_PRELOAD_PATH", DEFAULT_GCR_PRELOAD_PATH)

        engine = sgl.Engine(
            model_path=MODEL_PATH,
            random_seed=42,
            enable_gcr=True,
            dp_size=2,
        )

        _generate_and_check(engine)

        engine.gcr_suspend()
        engine.gcr_resume()

        _generate_and_check(engine)
        engine.shutdown()

    def test_engine_single_dp(self):
        os.environ.setdefault("GCR_PRELOAD_PATH", DEFAULT_GCR_PRELOAD_PATH)
        engine = sgl.Engine(
            model_path=MODEL_PATH,
            random_seed=42,
            enable_gcr=True,
        )

        _generate_and_check(engine)

        engine.gcr_suspend()
        engine.gcr_resume()

        _generate_and_check(engine)
        engine.shutdown()


class TestGcrPreloadGrpc(unittest.TestCase):
    """Test GCR preload via the gRPC launcher path (scheduler_launcher.py)."""

    def test_grpc_launcher(self):
        os.environ.setdefault("GCR_PRELOAD_PATH", DEFAULT_GCR_PRELOAD_PATH)

        server_args = sgl.ServerArgs(
            model_path=MODEL_PATH,
            random_seed=42,
            enable_gcr=True,
        )

        scheduler_info, port_args, scheduler_procs = launch_scheduler_process_only(
            server_args
        )

        try:
            pids = [p.pid for p in scheduler_procs]
        finally:
            for proc in scheduler_procs:
                os.kill(proc.pid, SIGTERM)
            for proc in scheduler_procs:
                proc.join(timeout=10)


if __name__ == "__main__":
    unittest.main()