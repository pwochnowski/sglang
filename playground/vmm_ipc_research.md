# Weight Buffer Transfer: SGLang IPC Research

Research into using VMM IPC for SGLang weight transfer, with verl as reference implementation.

## How SGLang transfers weights today

SGLang flattens model parameters into a single contiguous GPU buffer via `FlattenedTensorBucket` (`torch.cat` of all parameters viewed as `uint8`). This buffer is serialized with `ForkingPickler`, which calls `storage._share_cuda_()` -> `cudaIpcGetMemHandle` under the hood. The result is a ~200 byte base64 string containing the IPC handle + tensor metadata (shape, dtype, strides, storage offset). The actual weight data never leaves the GPU.

The consumer deserializes with `pickle.load()`, which calls `cudaIpcOpenMemHandle` to map the **same physical GPU memory** into the consumer's address space. The consumer then copies from that mapped buffer into its model parameters.

**verl difference**: verl pre-allocates a fixed-size buffer (`torch.empty` or VMM) and copies weights into it in buckets, rather than using `torch.cat`. This avoids allocating a new buffer per transfer and supports models larger than a single bucket. Metadata goes over ZMQ separately; only the bulk data uses IPC.

## Copy count

Both SGLang and verl perform exactly **two GPU copies** per transfer:

1. **Source -> shared buffer**: SGLang does this via `torch.cat` (allocates + copies into a new contiguous tensor). verl does it via `buffer[offset:offset+n].copy_()` into a pre-allocated buffer.
2. **Shared buffer -> model parameters**: The consumer copies from the IPC-mapped buffer into its own model weights.

The IPC transport between processes is **zero-copy** in both cases -- the consumer maps the same physical GPU memory. VMM does not add an extra copy; it replaces the IPC mechanism, not the data flow.

## Why the copy into a shared buffer is necessary

1. **Layout**: Model parameters are scattered across GPU memory in separate allocations. You can't export hundreds of individual IPC handles. Flattening into one buffer means one handle (or one fd).
2. **Ownership**: The training engine continues mutating parameters (next optimizer step). The copy decouples the transfer from training -- the producer can resume immediately after copying, while the consumer reads from a stable snapshot.

## Name/layout remapping

Megatron and vLLM/HuggingFace use different parameter names and layouts. Megatron fuses QKV into `linear_qkv.weight` and gate/up into `linear_fc1.weight`; HuggingFace splits them into `q_proj`, `k_proj`, `v_proj`, `gate_proj`, `up_proj`. This conversion happens **before** the IPC transfer -- by the time data reaches the shared buffer, tensors are already in HF format. The remapping uses tensor views (`.narrow()`, `.view()`), not new allocations.

**SGLang note**: SGLang receives weights already in HF naming. The remapping is the training framework's responsibility, not SGLang's.

## Buffer lifecycle

SGLang's approach is stateless per-transfer: `torch.cat` allocates a buffer, `ForkingPickler` exports the handle, the consumer maps it, both sides eventually release.

verl reuses the buffer across buckets within a transfer session (overwriting in-place with ZMQ handshakes as fences), then frees it at session end. The buffer is reallocated each session. This could be optimized to persist across sessions.

The torch consumer caches IPC mappings in `SharedCache` (`shared_cache` dict keyed by `(handle, offset)`). If the same handle arrives again, `cudaIpcOpenMemHandle` is not called a second time. The producer has no such cache -- `cudaIpcGetMemHandle` is called each time, though it returns the same handle for the same `cudaMalloc` block.

## The caching allocator concern

The torch `reduce_tensor` source contains a long comment about `cudaIpcGetMemHandle` operating on entire `cudaMalloc` blocks, not individual tensors within them. This means you might share more memory than intended, and the consumer must track offsets within larger allocations.

**This is moot for both SGLang and verl** because both flatten into a dedicated buffer first. The IPC handle refers to that buffer specifically, not to a sub-region of a larger caching allocator block.

**This concern does not apply to VMM at all** because `cuMemCreate` allocates exactly the requested size -- there is no caching allocator carving up larger blocks.

## Why `cudaIpcGetMemHandle` fails with VMM memory

`cudaIpcGetMemHandle` only works on memory allocated via `cudaMalloc`. Memory allocated via `cuMemCreate` + `cuMemMap` (VMM) or memory managed by certain allocator hooks/pools returns `cudaErrorInvalidValue`. This is the error observed in `test_cuda_ipc_compat.py`.

## VMM IPC as a replacement

VMM replaces the IPC handle mechanism, not the data flow:

| | SGLang today | VMM replacement |
|---|---|---|
| Allocate buffer | `torch.cat` / `torch.empty` (cudaMalloc) | `cuMemCreate` + `cuMemMap` |
| Export handle | `cudaIpcGetMemHandle` (via ForkingPickler) | `cuMemExportToShareableHandle` -> POSIX fd |
| Transport handle | pickle + base64 string over RPC | fd over Unix socket (`SCM_RIGHTS`) |
| Import handle | `cudaIpcOpenMemHandle` (via pickle.load) | `cuMemImportFromShareableHandle` + `cuMemMap` |
| Shared memory semantics | Both sides see same physical memory | Same -- both sides see same physical memory |
| Refcounting | CUDA runtime refcounts the mapping | CUDA driver refcounts the `CUmemGenericAllocationHandle` |
| Synchronization | Torch records a CUDA event at serialize time | Explicit `torch.cuda.synchronize()` before signaling consumer |

The only structural addition is the Unix domain socket for fd transport -- you can't embed a file descriptor in a string. Metadata (names, shapes, dtypes, offsets) still travels over the existing RPC/ZMQ channel.

## `vmm_ipc.py`

Standalone module with no framework imports. Dependencies: `cuda-python` (for `cuda.bindings.driver`), `torch`, stdlib. It provides:

- `alloc_vmm_buffer(size, device)` -> `VmmAllocation` with exported fd
- `import_vmm_buffer(fd, size, device)` -> `VmmAllocation` mapped in consumer
- `wrap_as_torch_uint8(alloc)` -> torch tensor view via DLPack (zero-copy)
- `free_vmm_buffer(alloc)` -> teardown in correct order
- `send_fd()` / `recv_fd()` -> fd transport over UDS via `SCM_RIGHTS`
- Sidecar UDS helpers for socket setup

## Reference files

- VMM IPC module: `vmm_ipc.py` (standalone, can be dropped into any project)
- Bucketed transfer (reference for all three IPC paths): `bucketed_weight_transfer.py`
- torch reduce_tensor (legacy IPC internals): `torch/multiprocessing/reductions.py`
