"""
Minimal reproduction of slime's colocated weight-sync IPC path.

Exercises the exact serialize → deserialize → copy chain that
update_weight_from_tensor.py uses:

  Producer (process A, GPU 0):
    1. Build named tensors on GPU
    2. FlattenedTensorBucket  → flatten into one contiguous GPU buffer
    3. MultiprocessingSerializer.serialize(…, output_str=True)
       → ForkingPickler → cudaIpcGetMemHandle → base64 string

  Consumer (process B, GPU 0):
    4. MultiprocessingSerializer.deserialize(blob)
       → cudaIpcOpenMemHandle → GPU tensor in consumer's address space
    5. FlattenedTensorBucket.reconstruct_tensors()  → recover named tensors
    6. .copy_() into consumer-owned destination (simulates SGLang weight load)
    7. Del the IPC-mapped tensors + torch.cuda.ipc_collect()

Run with:
    LD_PRELOAD=<your_hook.so> python tests/test_cuda_ipc_compat.py

If your LD_PRELOAD hook breaks the legacy cudaIpc* path you'll see errors
at step 3 (serialize) or step 4 (deserialize).
"""

import multiprocessing as mp
import sys
import traceback

import torch


# ---------------------------------------------------------------------------
# These are inlined so the test has zero sglang/slime deps.
# They are faithful copies of the real classes.
# ---------------------------------------------------------------------------

import io
import pickle

import pybase64
from multiprocessing.reduction import ForkingPickler


class MultiprocessingSerializer:
    """sglang.srt.utils.common.MultiprocessingSerializer (inlined)"""

    @staticmethod
    def serialize(obj, output_str: bool = False):
        buf = io.BytesIO()
        ForkingPickler(buf).dump(obj)
        buf.seek(0)
        output = buf.read()
        if output_str:
            output = pybase64.b64encode(output).decode("utf-8")
        return output

    @staticmethod
    def deserialize(data):
        if isinstance(data, str):
            data = pybase64.b64decode(data, validate=True)
        return pickle.Unpickler(io.BytesIO(data)).load()


class FlattenedTensorBucket:
    """sglang.srt.weight_sync.tensor_bucket.FlattenedTensorBucket (inlined)"""

    def __init__(self, named_tensors=None, flattened_tensor=None, metadata=None):
        if named_tensors is not None:
            self.metadata = []
            parts = []
            idx = 0
            for name, tensor in named_tensors:
                flat = tensor.flatten().view(torch.uint8)
                n = flat.numel()
                self.metadata.append(
                    {"name": name, "shape": tensor.shape, "dtype": tensor.dtype, "start": idx, "end": idx + n}
                )
                parts.append(flat)
                idx += n
            self.flattened_tensor = torch.cat(parts, dim=0)
        else:
            self.flattened_tensor = flattened_tensor
            self.metadata = metadata

    def get_flattened_tensor(self):
        return self.flattened_tensor

    def get_metadata(self):
        return self.metadata

    def reconstruct_tensors(self):
        out = []
        for m in self.metadata:
            t = self.flattened_tensor[m["start"] : m["end"]].view(m["dtype"]).reshape(m["shape"])
            out.append((m["name"], t))
        return out


# ---------------------------------------------------------------------------
# Producer / consumer logic
# ---------------------------------------------------------------------------


def producer(queue: mp.Queue, device: int = 0):
    """Simulates the Megatron actor side of _send_to_colocated_engine."""
    try:
        torch.cuda.set_device(device)

        # Step 1: Create some model-like tensors on GPU
        named_tensors = [
            ("model.layers.0.self_attn.q_proj.weight", torch.randn(512, 512, device="cuda", dtype=torch.bfloat16)),
            ("model.layers.0.self_attn.k_proj.weight", torch.randn(128, 512, device="cuda", dtype=torch.bfloat16)),
            ("model.layers.0.mlp.gate_proj.weight", torch.randn(1024, 512, device="cuda", dtype=torch.float16)),
        ]

        # Keep CPU ground-truth for verification
        ground_truth = [(n, t.cpu().clone()) for n, t in named_tensors]

        # Step 2: FlattenedTensorBucket (same as update_weight_from_tensor.py:236-241)
        bucket = FlattenedTensorBucket(named_tensors=named_tensors)
        metadata = bucket.get_metadata()
        flattened_tensor_data = {
            "flattened_tensor": bucket.get_flattened_tensor(),
            "metadata": metadata,
        }

        # Step 3: Serialize with CUDA IPC handles (the critical path)
        blob = MultiprocessingSerializer.serialize(flattened_tensor_data, output_str=True)

        print(f"[producer] serialized {len(blob)} chars (base64), "
              f"flattened tensor: {bucket.get_flattened_tensor().shape} uint8 on cuda:{device}")

        queue.put(("ok", blob, ground_truth))

    except Exception:
        traceback.print_exc()
        queue.put(("error", traceback.format_exc(), None))


def consumer(queue: mp.Queue, device: int = 0):
    """Simulates the SGLang engine side of update_weights_from_tensor."""
    try:
        torch.cuda.set_device(device)

        tag, blob, ground_truth = queue.get(timeout=30)
        if tag != "ok":
            print(f"[consumer] producer failed:\n{blob}")
            sys.exit(1)

        # Step 4: Deserialize — this is where cudaIpcOpenMemHandle happens
        flattened_tensor_data = MultiprocessingSerializer.deserialize(blob)

        flattened_tensor = flattened_tensor_data["flattened_tensor"]
        metadata = flattened_tensor_data["metadata"]
        print(f"[consumer] deserialized flattened tensor: {flattened_tensor.shape} "
              f"on {flattened_tensor.device}")

        assert flattened_tensor.is_cuda, "deserialized tensor should be on GPU"

        # Step 5: Reconstruct named tensors (same as SGLang's model_runner path)
        bucket = FlattenedTensorBucket(flattened_tensor=flattened_tensor, metadata=metadata)
        reconstructed = bucket.reconstruct_tensors()

        # Step 6: Copy into consumer-owned destination tensors (simulates weight load)
        for (name, src), (gt_name, gt_tensor) in zip(reconstructed, ground_truth):
            assert name == gt_name, f"name mismatch: {name} vs {gt_name}"
            dst = torch.empty_like(src)  # consumer-owned allocation
            dst.copy_(src)

            # Verify against ground truth
            if not torch.equal(dst.cpu(), gt_tensor):
                print(f"[consumer] MISMATCH on {name}")
                sys.exit(1)
            print(f"[consumer] {name}: shape={tuple(dst.shape)} dtype={dst.dtype} ✓")

        # Step 7: Release IPC mappings (same as update_weight_from_tensor.py:163-165)
        del flattened_tensor_data, flattened_tensor, bucket, reconstructed
        torch.cuda.ipc_collect()

        print("[consumer] all tensors verified, IPC handles released")

    except Exception:
        traceback.print_exc()
        sys.exit(1)


def main():
    mp.set_start_method("spawn", force=True)

    if not torch.cuda.is_available():
        print("CUDA not available, skipping")
        sys.exit(0)

    queue = mp.Queue()
    p = mp.Process(target=producer, args=(queue,))
    c = mp.Process(target=consumer, args=(queue,))

    p.start()
    c.start()
    p.join(timeout=60)
    c.join(timeout=60)

    if p.exitcode != 0 or c.exitcode != 0:
        print(f"FAIL  (producer={p.exitcode}, consumer={c.exitcode})")
        sys.exit(1)

    print("PASS")


if __name__ == "__main__":
    main()
