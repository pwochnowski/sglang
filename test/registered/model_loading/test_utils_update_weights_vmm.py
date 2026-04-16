import os
import unittest

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh

from sglang.srt.entrypoints.engine import Engine
from sglang.srt.weight_sync.utils import update_weights_vmm
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=60, suite="stage-b-test-1-gpu-large")

# Mirrors test_utils_update_weights.py, but drives the VMM transport path
# through weight_sync/utils.py::update_weights_vmm().
#
# OPT-125m: implements get_weights_by_name (needed for verification) and is
# ungated on HF.
MODEL = "facebook/opt-125m"
FC1_SHAPE = (3072, 768)
PARAM_NAME_TEMPLATE = "model.decoder.layers.{i}.fc1.weight"


def is_distributed_available():
    required_vars = ["RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT"]
    return all(var in os.environ for var in required_vars)


def setup_single_process_distributed():
    if not is_distributed_available():
        os.environ["RANK"] = "0"
        os.environ["WORLD_SIZE"] = "1"
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "12357"
        os.environ["LOCAL_RANK"] = "0"


class TestUtilsUpdateWeightsVMM(unittest.TestCase):
    """Exercise weight_sync/utils.py::update_weights_vmm() end-to-end.

    This is the distributed/device_mesh entry point a training process
    would call in SPMD style. The in-file test in
    test_update_weights_from_tensor_vmm.py builds the VMM request inline and
    skips gather_object + device_uuid mapping; this test covers those.
    """

    @classmethod
    def setUpClass(cls):
        cls.setup_distributed()
        cls.setup_test_engine()
        cls.setup_device_mesh()

    @classmethod
    def tearDownClass(cls):
        if getattr(cls, "engine", None) is not None:
            cls.engine.shutdown()
        if dist.is_initialized():
            dist.destroy_process_group()

    @classmethod
    def setup_distributed(cls):
        setup_single_process_distributed()
        if not dist.is_initialized():
            try:
                dist.init_process_group(
                    backend="nccl" if torch.cuda.is_available() else "gloo"
                )
            except Exception as e:
                raise unittest.SkipTest(f"Could not init distributed backend: {e}")

        cls.rank = dist.get_rank()
        cls.world_size = dist.get_world_size()
        if torch.cuda.is_available():
            torch.cuda.set_device(cls.rank % torch.cuda.device_count())

        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
        os.environ["NCCL_CUMEM_ENABLE"] = "0"
        os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "4"
        os.environ["CUDA_MODULE_LOADING"] = "AUTO"

    @classmethod
    def setup_test_engine(cls):
        if cls.rank == 0:
            cls.engine = Engine(
                model_path=MODEL,
                mem_fraction_static=0.3,
                enable_memory_saver=True,
                tp_size=cls.world_size,
                disable_cuda_graph=False,
            )
        else:
            cls.engine = None

    @classmethod
    def setup_device_mesh(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA not available for device mesh")
        cls.device_mesh_key = "tp"
        cls.mesh = init_device_mesh(
            "cuda", (cls.world_size,), mesh_dim_names=(cls.device_mesh_key,)
        )

    def create_test_params_batch(self, num_params=3):
        """Build (name, tensor) pairs with all 1.5s so we can verify the update."""
        param_names = [PARAM_NAME_TEMPLATE.format(i=i) for i in range(2, 2 + num_params)]
        tensors = [torch.full(FC1_SHAPE, 1.5, device="cuda") for _ in param_names]
        return list(zip(param_names, tensors))

    def test_utils_update_weights_vmm(self):
        params_batch = self.create_test_params_batch(num_params=3)

        update_weights_vmm(
            engine=self.engine,
            params_batch=params_batch,
            device_mesh_key=self.device_mesh_key,
            device_mesh=self.mesh,
        )

        # Verify weights were actually written to the model
        if self.rank == 0:
            for name, _ in params_batch:
                actual_values = torch.tensor(self.engine.get_weights_by_name(name))[
                    0, :5
                ]
                self.assertTrue(
                    torch.allclose(
                        actual_values, torch.tensor([1.5] * 5), atol=0.002
                    ),
                    f"{name=} {actual_values=}",
                )


if __name__ == "__main__":
    unittest.main()
