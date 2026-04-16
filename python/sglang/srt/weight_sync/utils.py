import logging
import os
import sys
import threading
import uuid
from typing import Optional

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor

from sglang.srt.entrypoints.engine import Engine
from sglang.srt.managers.io_struct import (
    UpdateWeightsFromTensorReqInput,
    UpdateWeightsFromTensorVMMReqInput,
)
from sglang.srt.model_executor.model_runner import LocalSerializedTensor
from sglang.srt.utils import MultiprocessingSerializer

logger = logging.getLogger(__name__)


async def update_weights(
    engine: Engine,
    params_batch: list[tuple[str, torch.Tensor]],
    device_mesh_key: str,
    device_mesh: DeviceMesh,
    load_format: Optional[str] = None,
):
    """
    Update weights for the inference engine.
    This function is designed to be stateless, so that the caller process could keep the stateful engine.
    Example Use Case:
        - Multiple Producer Process will call this function in a SPMD style

    Args:
        engine: The inference engine created by the caller process.
        params_batch: A list of (name, tensor) tuples. We batched the tensors to avoid the overhead of cpu call.
        device_mesh_key: The key of the device mesh. Typically "tp" or "infer_tp"
        device_mesh: The device mesh.
        load_format: The format of the weights.
    """
    infer_tp_size = device_mesh[device_mesh_key].mesh.size()[0]
    infer_tp_rank = device_mesh[device_mesh_key].get_local_rank()
    from sglang.srt.utils.patch_torch import monkey_patch_torch_reductions

    monkey_patch_torch_reductions()

    # [
    #   (name0, ipc_tensor0_tp0),
    #   (name1, ipc_tensor1_tp0),
    # ]
    named_tensors_batch = [
        (
            name,
            MultiprocessingSerializer.serialize(
                _preprocess_tensor_for_update_weights(tensor.detach())
            ),
        )
        for name, tensor in params_batch
    ]

    if infer_tp_rank == 0:
        gathered_serialized_batches = [None for _ in range(infer_tp_size)]
    else:
        gathered_serialized_batches = None

    # [
    #   [ (name0, ipc_tensor0_tp0), (name1, ipc_tensor1_tp0) ],
    #   [ (name0, ipc_tensor0_tp1), (name1, ipc_tensor1_tp1) ],
    # ]
    dist.gather_object(
        obj=named_tensors_batch,
        object_gather_list=gathered_serialized_batches,
        dst=device_mesh[device_mesh_key].mesh.tolist()[0],
        group=device_mesh[device_mesh_key].get_group(),
    )

    if infer_tp_rank == 0:
        # Use zip(*) to "transpose" the data structure.
        # After transpose, the data structure is like:
        # [
        #   ( (name0, ipc_tensor0_tp0), (name0, ipc_tensor0_tp1) ),
        #   ( (name1, ipc_tensor1_tp0), (name1, ipc_tensor1_tp1) ),
        # ]
        logical_tensors = zip(*gathered_serialized_batches, strict=True)

        named_tensors = [
            # [
            #   (name0, LocalSerializedTensor(values=[ipc_tensor0_tp0, ipc_tensor0_tp1])),
            #   (name1, LocalSerializedTensor(values=[ipc_tensor1_tp0, ipc_tensor1_tp1])),
            # ]
            (
                tensor_group[0][0],
                LocalSerializedTensor(
                    values=[rank_part[1] for rank_part in tensor_group]
                ),
            )
            for tensor_group in logical_tensors
        ]

        update_weights_request = UpdateWeightsFromTensorReqInput(
            serialized_named_tensors=[
                MultiprocessingSerializer.serialize(named_tensors)
                for _ in range(infer_tp_size)
            ],
            load_format=load_format,
        )

        return await engine.update_weights_from_tensor(update_weights_request)


def update_weights_vmm(
    engine: Engine,
    params_batch: list[tuple[str, torch.Tensor]],
    device_mesh_key: str,
    device_mesh: DeviceMesh,
):
    """
    Update weights via VMM IPC.  Stateless: allocates a VMM buffer per call,
    copies params in, sends the fd over UDS, and frees after the consumer
    has imported.

    Called in SPMD style by all TP ranks in the training process.
    Only rank 0 calls engine.update_weights_from_tensor_vmm().

    Args:
        engine: The inference engine (only used on rank 0).
        params_batch: A list of (name, tensor) tuples (full, unsharded).
        device_mesh_key: The key of the device mesh. Typically "tp" or "infer_tp".
        device_mesh: The device mesh.
    """
    from sglang.srt.weight_sync.vmm_ipc import (
        alloc_vmm_buffer,
        free_vmm_buffer,
        open_sidecar_listener,
        send_fd,
        wrap_as_torch_uint8,
    )

    tp_size = device_mesh[device_mesh_key].mesh.size()[0]
    tp_rank = device_mesh[device_mesh_key].get_local_rank()
    group = device_mesh[device_mesh_key].get_group()
    device = torch.cuda.current_device()

    # 1. Preprocess (gather DTensor shards if needed) and flatten to uint8
    processed = [
        (name, _preprocess_tensor_for_update_weights(t.detach()))
        for name, t in params_batch
    ]

    metadata = []
    flat_parts = []
    offset = 0
    for name, tensor in processed:
        flat = tensor.flatten().view(torch.uint8)
        metadata.append(
            {
                "name": name,
                "shape": list(tensor.shape),
                "dtype": str(tensor.dtype).replace("torch.", ""),
                "start_idx": offset,
                "end_idx": offset + flat.numel(),
            }
        )
        flat_parts.append(flat)
        offset += flat.numel()

    # 2. Allocate VMM buffer and copy flattened data in. The allocation
    #    is rounded up to VMM granularity (typically 2MB), so slice `buf`
    #    to exactly `offset` bytes before using it as a cat output.
    alloc = alloc_vmm_buffer(offset, device)
    buf = wrap_as_torch_uint8(alloc)
    torch.cat(flat_parts, out=buf[:offset])
    torch.cuda.synchronize(device)

    # 3. Start UDS listener in background thread
    uds_path = f"/tmp/sglang-vmm-{uuid.uuid4()}-tp{tp_rank}.sock"
    listener = open_sidecar_listener(uds_path)

    def _serve():
        conn, _ = listener.accept()
        send_fd(conn, alloc.fd)
        conn.close()
        listener.close()

    thread = threading.Thread(target=_serve, daemon=True)
    thread.start()

    # 4. Gather per-rank info to rank 0
    device_uuid = f"GPU-{torch.cuda.get_device_properties(device).uuid!s}"
    my_info = {
        "uds_path": uds_path,
        "device_uuid": device_uuid,
        "buffer_size": alloc.size,
    }
    gathered = [None] * tp_size if tp_rank == 0 else None
    dist.gather_object(
        obj=my_info,
        object_gather_list=gathered,
        dst=device_mesh[device_mesh_key].mesh.tolist()[0],
        group=group,
    )

    # 5. Rank 0 triggers the engine endpoint (blocks until all workers finish)
    if tp_rank == 0:
        req = UpdateWeightsFromTensorVMMReqInput(
            uds_paths={g["device_uuid"]: g["uds_path"] for g in gathered},
            buffer_sizes={g["device_uuid"]: g["buffer_size"] for g in gathered},
            tensor_metadata=metadata,
        )
        result = engine.update_weights_from_tensor_vmm(req)
        logger.info("VMM weight update result: %s", result)

    # 6. Cleanup -- thread should already be done since the worker connected
    #    during the engine call
    thread.join(timeout=30)
    rc = sys.getrefcount(buf)
    if rc > 2:  # getrefcount itself adds 1, so 2 means sole owner
        logger.warning("VMM buf refcount = %d before del (expected 2), "
                       "something still holds a reference", rc)
    del buf
    free_vmm_buffer(alloc, close_fd=True)
    try:
        os.unlink(uds_path)
    except FileNotFoundError:
        pass


def _preprocess_tensor_for_update_weights(tensor: torch.Tensor):
    """
    Preprocess the tensor for update weights.
    Example Use Case:
        - FSDP: we gather tensor by calling full_tensor in _preprocess_tensor_for_update_weights
        - Megatron: we do nothing here, assuming it is gathered when feed into this func

    Args:
        tensor: The tensor to be preprocessed.

    Returns:
        The full tensor if it is a DTensor, otherwise the original tensor.
    """
    if isinstance(tensor, DTensor):
        return tensor.full_tensor()
    return tensor
