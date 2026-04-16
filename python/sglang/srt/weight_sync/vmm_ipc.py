"""
VMM IPC helpers for CUDA Virtual Memory Management-backed buffer sharing.

Provides allocation, export/import, DLPack wrapping, and POSIX-fd transport
over Unix domain sockets (SCM_RIGHTS).  No verl imports; fully self-contained.
"""

from __future__ import annotations

import ctypes
import logging
import os
import socket
import struct
import time
from dataclasses import dataclass, field

import torch

from cuda.bindings import driver as cu
from torch.utils.dlpack import from_dlpack

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# cuda-python error checking
# ---------------------------------------------------------------------------


def _ck(result, what: str = ""):
    """Unwrap a cuda-python (CUresult, *outputs) tuple; raise on non-success."""
    if isinstance(result, tuple):
        err, *rest = result
    else:
        err, rest = result, []
    if err != cu.CUresult.CUDA_SUCCESS:
        _, name = cu.cuGetErrorName(err)
        if isinstance(name, bytes):
            name = name.decode()
        raise RuntimeError(f"CUDA VMM error [{what}]: {name}")
    if len(rest) == 1:
        return rest[0]
    return tuple(rest) if rest else None


# ---------------------------------------------------------------------------
# DLPack v0.8 ctypes definitions (ported from probe_dlpack_wrap.py)
# ---------------------------------------------------------------------------

kDLCUDA = 2
kDLUInt = 1


class DLDevice(ctypes.Structure):
    _fields_ = [
        ("device_type", ctypes.c_int),
        ("device_id", ctypes.c_int),
    ]


class DLDataType(ctypes.Structure):
    _fields_ = [
        ("code", ctypes.c_uint8),
        ("bits", ctypes.c_uint8),
        ("lanes", ctypes.c_uint16),
    ]


class DLTensor(ctypes.Structure):
    _fields_ = [
        ("data", ctypes.c_void_p),
        ("device", DLDevice),
        ("ndim", ctypes.c_int32),
        ("dtype", DLDataType),
        ("shape", ctypes.POINTER(ctypes.c_int64)),
        ("strides", ctypes.POINTER(ctypes.c_int64)),
        ("byte_offset", ctypes.c_uint64),
    ]


class DLManagedTensor(ctypes.Structure):
    pass


DLManagedTensorDeleter = ctypes.CFUNCTYPE(None, ctypes.POINTER(DLManagedTensor))

DLManagedTensor._fields_ = [
    ("dl_tensor", DLTensor),
    ("manager_ctx", ctypes.c_void_p),
    ("deleter", DLManagedTensorDeleter),
]

# Use c_void_p (not py_object!) for the capsule pointer.  py_object would
# increment the capsule's refcount inside capsule_dealloc, creating a
# 0→1→0 re-entrancy loop that stack-overflows into SIGSEGV.
PyCapsule_Destructor = ctypes.CFUNCTYPE(None, ctypes.c_void_p)

_PyCapsule_New = ctypes.pythonapi.PyCapsule_New
_PyCapsule_New.restype = ctypes.py_object
_PyCapsule_New.argtypes = [ctypes.c_void_p, ctypes.c_char_p, PyCapsule_Destructor]


# ---------------------------------------------------------------------------
# VmmAllocation dataclass
# ---------------------------------------------------------------------------


@dataclass
class VmmAllocation:
    """Represents a VMM-backed CUDA allocation with optional fd export."""

    va: int  # int(CUdeviceptr) from cuMemAddressReserve
    size: int  # rounded up to granularity
    mem_handle: object  # CUmemGenericAllocationHandle
    fd: int | None  # POSIX fd from cuMemExportToShareableHandle
    device: int
    _dlpack_refs: tuple | None = field(default=None, repr=False)


# ---------------------------------------------------------------------------
# VMM allocation / import / free
# ---------------------------------------------------------------------------


def get_vmm_granularity(device: int) -> int:
    """Query recommended allocation granularity for the given device."""
    prop = cu.CUmemAllocationProp()
    prop.type = cu.CUmemAllocationType.CU_MEM_ALLOCATION_TYPE_PINNED
    prop.location.type = cu.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
    prop.location.id = device
    prop.requestedHandleTypes = cu.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR
    granularity = _ck(
        cu.cuMemGetAllocationGranularity(
            prop,
            cu.CUmemAllocationGranularity_flags.CU_MEM_ALLOC_GRANULARITY_RECOMMENDED,
        ),
        "cuMemGetAllocationGranularity",
    )
    return int(granularity)


def alloc_vmm_buffer(requested_size: int, device: int) -> VmmAllocation:
    """Allocate a VMM buffer and export it as a POSIX fd.

    Steps: cuMemAddressReserve -> cuMemCreate -> cuMemMap -> cuMemSetAccess ->
    cuMemExportToShareableHandle.  On failure, partial state is rolled back.
    """
    granularity = get_vmm_granularity(device)
    size = ((requested_size + granularity - 1) // granularity) * granularity
    if size != requested_size:
        logger.info(
            "VMM alloc: rounded %d -> %d (granularity %d)",
            requested_size,
            size,
            granularity,
        )

    va = None
    mem_handle = None
    mapped = False
    fd = None

    prop = cu.CUmemAllocationProp()
    prop.type = cu.CUmemAllocationType.CU_MEM_ALLOCATION_TYPE_PINNED
    prop.location.type = cu.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
    prop.location.id = device
    prop.requestedHandleTypes = cu.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR

    try:
        va = _ck(cu.cuMemAddressReserve(size, granularity, 0, 0), "cuMemAddressReserve")
        mem_handle = _ck(cu.cuMemCreate(size, prop, 0), "cuMemCreate")
        _ck(cu.cuMemMap(va, size, 0, mem_handle, 0), "cuMemMap")
        mapped = True

        access_desc = cu.CUmemAccessDesc()
        access_desc.location.type = cu.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
        access_desc.location.id = device
        access_desc.flags = cu.CUmemAccess_flags.CU_MEM_ACCESS_FLAGS_PROT_READWRITE
        _ck(cu.cuMemSetAccess(va, size, [access_desc], 1), "cuMemSetAccess")

        fd = _ck(
            cu.cuMemExportToShareableHandle(
                mem_handle,
                cu.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR,
                0,
            ),
            "cuMemExportToShareableHandle",
        )
    except Exception:
        if mapped:
            cu.cuMemUnmap(va, size)
        if va is not None:
            cu.cuMemAddressFree(va, size)
        if mem_handle is not None:
            cu.cuMemRelease(mem_handle)
        if fd is not None:
            os.close(fd)
        raise

    return VmmAllocation(
        va=int(va), size=size, mem_handle=mem_handle, fd=int(fd), device=device
    )


def import_vmm_buffer(fd: int, size: int, device: int) -> VmmAllocation:
    """Import a VMM buffer from a POSIX fd received via SCM_RIGHTS.

    The *size* must be the already-rounded value sent by the allocator side.
    """
    va = None
    mem_handle = None
    mapped = False

    try:
        mem_handle = _ck(
            cu.cuMemImportFromShareableHandle(
                fd,
                cu.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR,
            ),
            "cuMemImportFromShareableHandle",
        )

        granularity = get_vmm_granularity(device)
        va = _ck(
            cu.cuMemAddressReserve(size, granularity, 0, 0), "cuMemAddressReserve"
        )
        _ck(cu.cuMemMap(va, size, 0, mem_handle, 0), "cuMemMap")
        mapped = True

        access_desc = cu.CUmemAccessDesc()
        access_desc.location.type = cu.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
        access_desc.location.id = device
        access_desc.flags = cu.CUmemAccess_flags.CU_MEM_ACCESS_FLAGS_PROT_READWRITE
        _ck(cu.cuMemSetAccess(va, size, [access_desc], 1), "cuMemSetAccess")
    except Exception:
        if mapped:
            cu.cuMemUnmap(va, size)
        if va is not None:
            cu.cuMemAddressFree(va, size)
        if mem_handle is not None:
            cu.cuMemRelease(mem_handle)
        raise

    return VmmAllocation(
        va=int(va), size=size, mem_handle=mem_handle, fd=fd, device=device
    )


def wrap_as_torch_uint8(alloc: VmmAllocation) -> torch.Tensor:
    """Wrap a VmmAllocation as a 1-D torch.uint8 tensor via DLPack.

    The caller MUST:
    1. Drop the returned tensor BEFORE calling free_vmm_buffer.
    2. NOT drop the VmmAllocation until after the tensor is gone.

    DLPack refs (ctypes structs and callbacks) are stored on
    ``alloc._dlpack_refs`` so they outlive both the base tensor and any
    views of it.  The caller keeps ``alloc`` alive until after
    ``free_vmm_buffer``, which deliberately does NOT clear ``_dlpack_refs``
    — ensuring the DLPack deleter callback is always valid when
    ``StorageImpl::~StorageImpl()`` fires, regardless of tensor/view
    destruction order.
    """
    shape = (ctypes.c_int64 * 1)(alloc.size)

    managed = DLManagedTensor()
    managed.dl_tensor.data = ctypes.c_void_p(alloc.va)
    managed.dl_tensor.device = DLDevice(kDLCUDA, alloc.device)
    managed.dl_tensor.ndim = 1
    managed.dl_tensor.dtype = DLDataType(kDLUInt, 8, 1)
    managed.dl_tensor.shape = shape
    managed.dl_tensor.strides = ctypes.POINTER(ctypes.c_int64)()  # NULL -> compact
    managed.dl_tensor.byte_offset = 0
    managed.manager_ctx = None

    @DLManagedTensorDeleter
    def _deleter(_self_ptr):
        pass  # No-op: buffer lifetime is managed by free_vmm_buffer

    managed.deleter = _deleter

    @PyCapsule_Destructor
    def _capsule_destructor(_capsule):
        pass  # Only fires if capsule is destroyed without from_dlpack consuming it

    capsule = _PyCapsule_New(
        ctypes.addressof(managed),
        b"dltensor",
        _capsule_destructor,
    )
    # Store on alloc — alloc outlives buf (and any views) in both
    # producer (utils.py) and consumer (model_runner.py) call sites.
    alloc._dlpack_refs = (managed, shape, _deleter, _capsule_destructor)

    tensor = from_dlpack(capsule)
    # Destroy the consumed PyCapsule immediately.  Its destructor calls
    # _capsule_destructor, which must still be alive (guaranteed because
    # it's stored on alloc._dlpack_refs above).
    del capsule
    return tensor


def free_vmm_buffer(alloc: VmmAllocation, *, close_fd: bool) -> None:
    """Free a VmmAllocation.  Caller MUST have already dropped the wrapped tensor.

    Order: cuMemUnmap -> cuMemAddressFree -> cuMemRelease -> os.close(fd).
    """
    _ck(cu.cuMemUnmap(alloc.va, alloc.size), "cuMemUnmap")
    _ck(cu.cuMemAddressFree(alloc.va, alloc.size), "cuMemAddressFree")
    _ck(cu.cuMemRelease(alloc.mem_handle), "cuMemRelease")
    if close_fd and alloc.fd is not None:
        os.close(alloc.fd)


# ---------------------------------------------------------------------------
# Sidecar UDS for fd transport via SCM_RIGHTS
# ---------------------------------------------------------------------------


def sidecar_path_for(zmq_handle: str) -> str:
    """Derive a UDS path for fd transport from the ZMQ IPC handle.

    ipc:///tmp/rl-colocate-zmq-<uuid>.sock -> /tmp/rl-colocate-fd-<uuid>.sock
    """
    path = zmq_handle.removeprefix("ipc://")
    return path.replace("-zmq-", "-fd-")


def open_sidecar_listener(path: str) -> socket.socket:
    """Bind a AF_UNIX SOCK_STREAM listener.  Unlinks stale socket first."""
    try:
        os.unlink(path)
    except FileNotFoundError:
        pass
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.bind(path)
    sock.listen(1)
    return sock


def open_sidecar_client(path: str, retries: int = 10) -> socket.socket:
    """Connect to the sidecar listener with retries on ConnectionRefusedError."""
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    for attempt in range(retries):
        try:
            sock.connect(path)
            return sock
        except (ConnectionRefusedError, FileNotFoundError):
            if attempt == retries - 1:
                sock.close()
                raise
            time.sleep(0.1 * (attempt + 1))
    sock.close()
    raise RuntimeError(f"Failed to connect to {path} after {retries} retries")


def send_fd(sock: socket.socket, fd: int) -> None:
    """Send a file descriptor over a Unix domain socket via SCM_RIGHTS."""
    sock.sendmsg(
        [b"\x00"],
        [(socket.SOL_SOCKET, socket.SCM_RIGHTS, struct.pack("i", fd))],
    )


def recv_fd(sock: socket.socket) -> int:
    """Receive a file descriptor from a Unix domain socket via SCM_RIGHTS."""
    msg, ancdata, flags, addr = sock.recvmsg(
        1, socket.CMSG_SPACE(struct.calcsize("i"))
    )
    for cmsg_level, cmsg_type, cmsg_data in ancdata:
        if cmsg_level == socket.SOL_SOCKET and cmsg_type == socket.SCM_RIGHTS:
            return struct.unpack("i", cmsg_data[: struct.calcsize("i")])[0]
    raise RuntimeError("No file descriptor received via SCM_RIGHTS")
