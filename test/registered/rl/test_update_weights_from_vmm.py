from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=240, suite="stage-b-test-1-gpu-small")

import gc
import json
import os
import random
import threading
import time
import unittest
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager

import requests
import torch

import sglang as sgl
from sglang.srt.managers.io_struct import UpdateWeightsFromTensorVMMReqInput
from sglang.srt.utils import kill_process_tree
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

# OPT-125m: hidden_size=768, ffn_dim=3072, num_layers=12.
# Chosen because it implements get_weights_by_name (Qwen2 doesn't) and is
# ungated on HF (Llama is gated).
MODEL = "facebook/opt-125m"
FC1_SHAPE = (3072, 768)
PARAM_NAME_TEMPLATE = "model.decoder.layers.{i}.fc1.weight"
PARAM_LAYER_RANGE = range(2, 10)


def _check_param(engine, param_name, expect_values):
    actual_values = torch.tensor(engine.get_weights_by_name(param_name))[0, :5]
    assert torch.allclose(
        actual_values, torch.tensor(expect_values), atol=0.002
    ), f"{actual_values=}"


@contextmanager
def _vmm_producer(param_names, new_tensor, device):
    """Producer-side VMM setup for single-process testing.

    Allocates a VMM buffer, flattens the given tensor repeated once per
    param name into it, and serves the fd over a UDS listener thread.
    Yields a dict matching ``UpdateWeightsFromTensorVMMReqInput`` fields.
    Cleans up on exit regardless of whether the consumer connected.
    """
    from sglang.srt.weight_sync.vmm_ipc import (
        alloc_vmm_buffer,
        free_vmm_buffer,
        open_sidecar_listener,
        send_fd,
        wrap_as_torch_uint8,
    )

    # Flatten all params into a single uint8 buffer
    flat_parts = []
    metadata = []
    offset = 0
    for name in param_names:
        flat = new_tensor.detach().flatten().view(torch.uint8)
        metadata.append(
            {
                "name": name,
                "shape": list(new_tensor.shape),
                "dtype": str(new_tensor.dtype).replace("torch.", ""),
                "start_idx": offset,
                "end_idx": offset + flat.numel(),
            }
        )
        flat_parts.append(flat)
        offset += flat.numel()

    # VMM rounds allocation up to granularity; slice `buf` to exact size.
    alloc = alloc_vmm_buffer(offset, device)
    buf = wrap_as_torch_uint8(alloc)
    torch.cat(flat_parts, out=buf[:offset])
    torch.cuda.synchronize(device)

    uds_path = f"/tmp/sglang-vmm-test-{uuid.uuid4()}.sock"
    listener = open_sidecar_listener(uds_path)

    def _serve():
        try:
            conn, _ = listener.accept()
            send_fd(conn, alloc.fd)
            conn.close()
        finally:
            listener.close()

    thread = threading.Thread(target=_serve, daemon=True)
    thread.start()

    device_uuid = f"GPU-{torch.cuda.get_device_properties(device).uuid!s}"
    req_fields = {
        "uds_paths": {device_uuid: uds_path},
        "buffer_sizes": {device_uuid: alloc.size},
        "tensor_metadata": metadata,
    }

    try:
        yield req_fields
    finally:
        thread.join(timeout=30)
        del buf
        free_vmm_buffer(alloc, close_fd=True)
        try:
            os.unlink(uds_path)
        except FileNotFoundError:
            pass


def _vmm_update_weights_engine(engine, param_names, new_tensor, device):
    """Run VMM update via the in-process Engine API."""
    with _vmm_producer(param_names, new_tensor, device) as fields:
        req = UpdateWeightsFromTensorVMMReqInput(**fields)
        return engine.update_weights_from_tensor_vmm(req)


def test_update_weights_from_tensor_vmm(tp_size):
    assert torch.cuda.device_count() >= tp_size, f"At least {tp_size} GPUs are required"
    torch.cuda.empty_cache()

    engine = sgl.Engine(model_path=MODEL, tp_size=tp_size)

    param_names = [PARAM_NAME_TEMPLATE.format(i=i) for i in PARAM_LAYER_RANGE]

    # Snapshot original values -- we'll verify they change
    original_values = torch.tensor(engine.get_weights_by_name(param_names[0]))[0, :5]

    memory_before = torch.cuda.memory_allocated()
    device = torch.cuda.current_device()
    new_tensor = torch.full(FC1_SHAPE, 1.5, device=f"cuda:{device}")

    time_start = time.perf_counter()
    result = _vmm_update_weights_engine(engine, param_names, new_tensor, device)
    elapsed = time.perf_counter() - time_start
    print(f"VMM update time: {elapsed:.03f}s, result: {result}")

    # Verify weights were updated
    for param_name in param_names[:3]:
        _check_param(engine, param_name, [1.5] * 5)

    # Sanity-check: the weights we just wrote differ from the originals
    assert not torch.allclose(
        original_values, torch.tensor([1.5] * 5), atol=0.002
    ), "Original weights already matched 1.5 -- test is not meaningful"

    engine.shutdown()

    del new_tensor
    gc.collect()
    torch.cuda.empty_cache()
    memory_after = torch.cuda.memory_allocated()
    assert (
        memory_after <= memory_before + 1024
    ), f"Memory leak detected: {memory_after - memory_before} bytes"


class TestUpdateWeightsFromTensorVMM(CustomTestCase):
    def test_update_weights_from_tensor_vmm_tp1(self):
        test_update_weights_from_tensor_vmm(tp_size=1)

    def test_update_weights_from_tensor_vmm_tp2(self):
        if torch.cuda.device_count() < 2:
            self.skipTest("Need at least 2 GPUs for tp_size=2")
        test_update_weights_from_tensor_vmm(tp_size=2)


class TestServerUpdateWeightsFromTensorVMMNonBlocking(CustomTestCase):
    """Exercise the /update_weights_from_tensor_vmm HTTP endpoint with the
    tokenizer_manager lock contended by in-flight decode requests.

    Mirrors TestServerUpdateWeightsFromTensorNonBlocking from
    test_update_weights_from_tensor.py but drives the VMM transport.
    """

    @classmethod
    def setUpClass(cls):
        cls.model = MODEL
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=["--max-running-requests", 8],
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def run_decode(self, max_new_tokens=32):
        response = requests.post(
            self.base_url + "/generate",
            json={
                "text": f"Question: {random.randint(0, 100)},The capital of France is",
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": max_new_tokens,
                    "ignore_eos": True,
                },
            },
        )
        return response.json()

    def pause_generation(self, mode):
        return requests.post(
            self.base_url + "/pause_generation", json={"mode": mode}
        ).json()

    def continue_generation(self):
        return requests.post(self.base_url + "/continue_generation", json={}).json()

    def run_update_weights_vmm(self, req_fields, flush_cache=True):
        response = requests.post(
            self.base_url + "/update_weights_from_tensor_vmm",
            json={**req_fields, "flush_cache": flush_cache},
        )
        return response.json()

    def test_update_weights(self):
        num_requests = 32
        device = torch.cuda.current_device()
        param_names = [PARAM_NAME_TEMPLATE.format(i=i) for i in PARAM_LAYER_RANGE]
        new_tensor = torch.full(FC1_SHAPE, 1.5, device=f"cuda:{device}")

        with ThreadPoolExecutor(num_requests) as executor:
            futures = [
                executor.submit(self.run_decode, 3000) for _ in range(num_requests)
            ]

            # ensure decodes have started
            time.sleep(2)

            # abort mode -- server becomes idle before /update_weights returns
            self.pause_generation("abort")
            with _vmm_producer(param_names, new_tensor, device) as fields:
                ret = self.run_update_weights_vmm(fields, flush_cache=True)
            self.assertTrue(ret["success"], msg=json.dumps(ret))
            self.continue_generation()

            # requests were aborted by pause_generation("abort"); drain futures
            for future in as_completed(futures):
                future.result()

        # Verify weights updated by querying through the HTTP endpoint
        for param_name in param_names[:3]:
            response = requests.post(
                self.base_url + "/get_weights_by_name",
                json={"name": param_name},
            )
            actual_values = torch.tensor(response.json())[0, :5]
            assert torch.allclose(
                actual_values, torch.tensor([1.5] * 5), atol=0.002
            ), f"{param_name=} {actual_values=}"


if __name__ == "__main__":
    unittest.main()
