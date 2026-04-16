# SLIME VMM IPC Integration

How to wire SLIME's weight update path through the new VMM IPC endpoint, and how to eliminate remaining legacy CUDA IPC (`ForkingPickler` / `cudaIpcGetMemHandle`) usage across SGLang.

## Current state

### What's already done (this branch)

The VMM IPC endpoint is fully implemented end-to-end:

| Layer | File | What exists |
|-------|------|-------------|
| Low-level VMM primitives | `python/sglang/srt/weight_sync/vmm_ipc.py` | `alloc_vmm_buffer`, `import_vmm_buffer`, `wrap_as_torch_uint8`, `free_vmm_buffer`, `send_fd`/`recv_fd` over UDS |
| Producer-side helper | `python/sglang/srt/weight_sync/utils.py` :: `update_weights_vmm()` | Flatten params -> VMM buffer -> UDS listener -> gather metadata -> `engine.update_weights_from_tensor_vmm()` |
| Request dataclass | `python/sglang/srt/managers/io_struct.py` :: `UpdateWeightsFromTensorVMMReqInput` | `uds_paths`, `buffer_sizes`, `tensor_metadata`, `flush_cache` |
| Engine API | `python/sglang/srt/entrypoints/engine.py` :: `update_weights_from_tensor_vmm()` | Routes to tokenizer_manager |
| HTTP endpoint | `python/sglang/srt/entrypoints/http_server.py` :: `POST /update_weights_from_tensor_vmm` | FastAPI route |
| Scheduler dispatch | `python/sglang/srt/managers/scheduler.py` + `scheduler_update_weights_mixin.py` | Dispatch + handler |
| TP worker passthrough | `python/sglang/srt/managers/tp_worker.py` | Delegates to model_runner |
| Consumer logic | `python/sglang/srt/model_executor/model_runner.py` :: `update_weights_from_tensor_vmm()` | recv_fd -> import -> reconstruct tensors -> load_weights -> sync -> free |
| Tests | `test/registered/rl/test_update_weights_from_tensor_vmm.py`, `test/registered/model_loading/test_utils_update_weights_vmm.py` | Unit + integration |

### What SLIME does today

SLIME is an external framework (not in this repo) that calls SGLang's weight update API. Its flow:

```
1. Upload Megatron weights from CPU to GPU in buckets (prevents OOM for large MoE)
2. Broadcast across PP/EP ranks
3. all_gather + torch.cat across TP ranks -> full tensor
4. convert_to_hf() -> HF-named tensors
5. MultiprocessingSerializer.serialize(converted_named_tensors, output_str=True)
     -> ForkingPickler -> cudaIpcGetMemHandle -> base64 string
6. dist.gather_object(ipc_handle, ...) to gather src rank
7. gather src calls engine.update_weights_from_tensor.remote(ipc_handles=...)
8. SGLang side: MultiprocessingSerializer.deserialize -> cudaIpcOpenMemHandle
     -> reconstruct tensor -> model.load_weights()
```

Steps 5-8 are the legacy CUDA IPC path. This fails on VMM-allocated memory and is what we want to replace.

## SLIME integration plan

### What changes in SLIME (external repo)

Replace `_update_converted_params_from_tensor()` with a VMM-based equivalent. The new flow:

```python
def _update_converted_params_from_tensor_vmm(self, converted_named_tensors):
    from sglang.srt.weight_sync.vmm_ipc import (
        alloc_vmm_buffer, wrap_as_torch_uint8, free_vmm_buffer,
        open_sidecar_listener, send_fd,
    )
    import threading, uuid

    device = torch.cuda.current_device()

    # 1. Flatten tensors to uint8
    metadata = []
    flat_parts = []
    offset = 0
    for name, tensor in converted_named_tensors:
        flat = tensor.flatten().view(torch.uint8)
        metadata.append({
            "name": name,
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype).replace("torch.", ""),
            "start_idx": offset,
            "end_idx": offset + flat.numel(),
        })
        flat_parts.append(flat)
        offset += flat.numel()

    # 2. Allocate VMM buffer and copy
    alloc = alloc_vmm_buffer(offset, device)
    buf = wrap_as_torch_uint8(alloc)
    torch.cat(flat_parts, out=buf[:offset])
    torch.cuda.synchronize(device)

    # 3. UDS listener for fd transport
    uds_path = f"/tmp/sglang-vmm-{uuid.uuid4()}.sock"
    listener = open_sidecar_listener(uds_path)
    def _serve():
        conn, _ = listener.accept()
        send_fd(conn, alloc.fd)
        conn.close()
        listener.close()
    thread = threading.Thread(target=_serve, daemon=True)
    thread.start()

    # 4. Gather per-rank info to gather src
    device_uuid = f"GPU-{torch.cuda.get_device_properties(device).uuid!s}"
    my_info = {"uds_path": uds_path, "device_uuid": device_uuid, "buffer_size": alloc.size}
    ipc_infos = (
        [None] * dist.get_world_size(self._ipc_gather_group)
        if self._ipc_gather_src == dist.get_rank() else None
    )
    dist.gather_object(my_info, object_gather_list=ipc_infos,
                       dst=self._ipc_gather_src, group=self._ipc_gather_group)

    # 5. Gather src calls VMM endpoint
    if dist.get_rank() == self._ipc_gather_src:
        from sglang.srt.managers.io_struct import UpdateWeightsFromTensorVMMReqInput
        req = UpdateWeightsFromTensorVMMReqInput(
            uds_paths={g["device_uuid"]: g["uds_path"] for g in ipc_infos},
            buffer_sizes={g["device_uuid"]: g["buffer_size"] for g in ipc_infos},
            tensor_metadata=metadata,
        )
        # Ray remote call to SGLang engine
        ref = self._ipc_engine.update_weights_from_tensor_vmm.remote(req)
        ray.get(ref)

    # 6. Cleanup
    thread.join(timeout=30)
    del buf
    free_vmm_buffer(alloc, close_fd=True)
    try:
        os.unlink(uds_path)
    except FileNotFoundError:
        pass

    converted_named_tensors.clear()
    torch.cuda.empty_cache()
```

Key differences from the old path:
- No `MultiprocessingSerializer.serialize()` (no `ForkingPickler`, no `cudaIpcGetMemHandle`)
- Tensor data goes into a VMM buffer instead of being IPC-handle-serialized
- fd transport over UDS replaces base64 IPC handle strings over Ray/gather
- Calls `update_weights_from_tensor_vmm` instead of `update_weights_from_tensor`

### What changes in SGLang (this repo)

Nothing in the core VMM endpoint -- it's already done. But SLIME calls the engine via Ray `.remote()`, so we need to verify the Ray actor wrapper exposes `update_weights_from_tensor_vmm`. Check whatever actor class wraps `Engine` in SLIME's integration layer.

### Bucket compatibility

SLIME processes weights in buckets (a subset of parameters per iteration). The VMM path is fully compatible: each bucket call allocates a fresh VMM buffer, transfers, and frees. No persistent state between buckets. The `flush_cache` flag should be `True` only on the last bucket.

## Legacy IPC paths inventory

All places in SGLang that use `MultiprocessingSerializer` or `ForkingPickler` for CUDA tensor sharing:

### 1. `update_weights_from_tensor` path (SLIME + verl primary path)

**Producer side:**

| Location | What it does | VMM replacement |
|----------|-------------|-----------------|
| `weight_sync/utils.py` :: `update_weights()` | `MultiprocessingSerializer.serialize()` each tensor, gather handles, serialize `named_tensors` again for each TP rank | Already replaced by `update_weights_vmm()` in same file |
| `entrypoints/engine.py` :: `update_weights_from_tensor()` | `MultiprocessingSerializer.serialize(named_tensors)` per TP rank | Bypassed when caller uses `update_weights_from_tensor_vmm()` instead |
| `entrypoints/http_server_engine.py` :: out-of-process engine | Same as engine.py but over HTTP | Bypassed when using VMM HTTP endpoint |

**Consumer side:**

| Location | What it does | VMM replacement |
|----------|-------------|-----------------|
| `model_runner.py` :: `update_weights_from_tensor()` | Calls `_unwrap_tensor()` -> `LocalSerializedTensor.get()` -> `MultiprocessingSerializer.deserialize()` -> `cudaIpcOpenMemHandle` | Bypassed: `update_weights_from_tensor_vmm()` reconstructs tensors directly from VMM buffer |
| `model_runner.py` :: `LocalSerializedTensor` dataclass | Wraps per-rank serialized bytes | Not needed for VMM path |
| `utils/patch_torch.py` :: `monkey_patch_torch_reductions()` | Monkey-patches `reduce_tensor`/`rebuild_cuda_tensor` to use device UUIDs instead of ordinals (workaround for multi-GPU IPC handle bug) | Not needed for VMM path (VMM uses device UUID natively in `uds_paths` dict) |

### 2. `update_weights_from_ipc` path (checkpoint-engine)

| Location | What it does |
|----------|-------------|
| `entrypoints/engine.py` :: `update_weights_from_ipc()` | Takes ZMQ handles dict, routes to scheduler |
| `model_runner.py` :: `update_weights_from_ipc()` | Delegates to `checkpoint_engine_worker.py` |
| `checkpoint_engine/checkpoint_engine_worker.py` | Imports `checkpoint_engine.worker.update_weights_from_ipc` (external package), calls it with ZMQ context |
| `checkpoint_engine/update.py` | HTTP client posting to `/update_weights_from_ipc` |

This path uses ZMQ for handle transport, not `ForkingPickler` directly. The external `checkpoint-engine` package does the actual IPC. **Migration**: This is a separate concern from SLIME. The checkpoint-engine could add a VMM backend, but it has its own IPC mechanism already. Lower priority.

### 3. Speculative decoding (EAGLE workers)

| Location | What it does |
|----------|-------------|
| `speculative/eagle_worker.py` :: `update_weights_from_tensor()` | `MultiprocessingSerializer.deserialize(recv_req.serialized_named_tensors[self.tp_rank])` |
| `speculative/eagle_worker_v2.py` :: `update_weights_from_tensor()` | Same |

These consume the same `UpdateWeightsFromTensorReqInput`. When EAGLE workers receive weight updates, they deserialize IPC handles. **Migration**: Add `update_weights_from_tensor_vmm()` to EAGLE workers, mirroring `model_runner.py`'s implementation. The EAGLE worker's model_runner already has `update_weights_from_tensor_vmm()`, so the worker just needs a passthrough.

### 4. LoRA adapter loading

| Location | What it does |
|----------|-------------|
| `entrypoints/engine.py` :: `load_lora_adapter_from_tensors()` | `MultiprocessingSerializer.serialize(tensors, output_str=True)` |

This serializes LoRA tensors as base64 strings for transport. LoRA tensors are small (MBs), so the IPC handle approach is actually fine here -- the overhead is negligible and the tensors fit in CPU memory. **Migration**: Optional, low priority.

### 5. NaiveDistributed (disaggregated inference)

| Location | What it does |
|----------|-------------|
| `distributed/naive_distributed.py` :: `reduce_scatter_tensor()` | `MultiprocessingSerializer.serialize/deserialize` for tensor scatter |

Used for disaggregated inference tensor transport. **Migration**: Separate concern, not related to weight updates.

### 6. CUDA IPC transport utils (multimodal)

| Location | What it does |
|----------|-------------|
| `utils/cuda_ipc_transport_utils.py` | `storage._share_cuda_()` for multimodal tensor proxy |
| `managers/mm_utils.py` | Similar CUDA IPC for multimodal data |

These are for multimodal input tensor sharing, not weight updates. **Migration**: Out of scope for SLIME integration.

### 7. Custom all-reduce

| Location | What it does |
|----------|-------------|
| `distributed/device_communicators/custom_all_reduce.py` | CUDA IPC for all-reduce buffer sharing |

Infrastructure-level IPC, not weight updates. **Migration**: Out of scope.

## Priority order for legacy IPC elimination

### Must do (SLIME integration)

1. **SLIME caller code** (external repo): Replace `_update_converted_params_from_tensor` with VMM equivalent as shown above. This is the primary deliverable.

2. **verl caller code** (external): verl's `fsdp_sglang.py` uses `update_weights()` from `weight_sync/utils.py`. It can switch to `update_weights_vmm()` from the same module. No SGLang changes needed.

### Should do (eliminate legacy path for weight updates)

3. **EAGLE workers**: Add `update_weights_from_tensor_vmm()` passthrough. ~5 lines each in `eagle_worker.py` and `eagle_worker_v2.py`.

4. **Deprecate `update_weights_from_tensor` path**: Once all callers (SLIME, verl) have migrated, the entire `update_weights_from_tensor` path can be deprecated:
   - `weight_sync/utils.py` :: `update_weights()`
   - `model_runner.py` :: `update_weights_from_tensor()`, `_unwrap_tensor()`, `LocalSerializedTensor`
   - `engine.py` :: `update_weights_from_tensor()`
   - `utils/patch_torch.py` :: `monkey_patch_torch_reductions()` (the reduce_tensor patches become unnecessary)
   - `UpdateWeightsFromTensorReqInput` dataclass
   - All the communicator/dispatcher/scheduler plumbing for this request type

### Nice to have (separate scope)

5. **Checkpoint-engine IPC path**: Has its own ZMQ-based mechanism. Could add VMM backend but works fine as-is.

6. **LoRA, NaiveDistributed, multimodal IPC, custom all-reduce**: Not weight update paths. Leave as-is.

## What does NOT change

- `model.load_weights()` -- final `copy_()` into model parameters is the same regardless of IPC mechanism
- `FlattenedTensorBucket` -- still useful for tensor flattening logic, though VMM path does its own flattening
- Name/layout remapping -- still the training framework's responsibility, happens before IPC
- The two-copy data flow -- producer copies into shared buffer, consumer copies into model params
