"""
Tests for VMM DLPack lifetime management.

Reproduces the segfault pattern from model_runner.update_weights_from_tensor_vmm:
views of a VMM-backed tensor must not outlive the DLPack ctypes callback refs.

Run with: pytest test/unit/test_vmm_dlpack_lifetime.py -v -s
Requires: CUDA device + cuda-python
"""

import multiprocessing
import os
import sys

import pytest

try:
    import torch

    _HAS_CUDA = torch.cuda.is_available()
except ImportError:
    _HAS_CUDA = False


# ---------------------------------------------------------------------------
# Subprocess helpers — segfaults kill the process, so we isolate each case
# ---------------------------------------------------------------------------


def _log(msg):
    sys.stderr.write(f"[vmm_test] {msg}\n")
    sys.stderr.flush()


def _run_in_subprocess(fn, timeout=30):
    """Run fn in a spawned subprocess; return (exitcode, error_msg_or_None)."""
    ctx = multiprocessing.get_context("spawn")
    q = ctx.Queue()
    p = ctx.Process(target=_subprocess_wrapper, args=(fn, q))
    p.start()
    p.join(timeout=timeout)
    if p.is_alive():
        p.terminate()
        p.join(5)
        return -1, "timed out"
    import queue as queue_mod

    try:
        status, err = q.get(timeout=5)
    except queue_mod.Empty:
        return p.exitcode, f"no result (exitcode={p.exitcode}, likely segfault)"
    if status != "ok":
        return 1, err
    return 0, None


def _subprocess_wrapper(fn, result_queue):
    try:
        fn()
        result_queue.put(("ok", None))
    except Exception:
        import traceback

        tb = traceback.format_exc()
        _log(f"EXCEPTION: {tb}")
        result_queue.put(("error", tb))
    finally:
        result_queue.close()
        result_queue.join_thread()
        os._exit(0)


# ---------------------------------------------------------------------------
# Test scenarios — each is a standalone function run in a subprocess
# ---------------------------------------------------------------------------


def _scenario_model_runner_pattern():
    """
    Exact model_runner.update_weights_from_tensor_vmm cleanup pattern.
    Views are deleted before the base tensor, alloc outlives everything.
    """
    import sys

    import torch

    _log("importing vmm_ipc")
    from sglang.srt.weight_sync.vmm_ipc import (
        alloc_vmm_buffer,
        free_vmm_buffer,
        wrap_as_torch_uint8,
    )

    _log("CUDA init")
    torch.cuda.init()
    torch.cuda.set_device(0)
    _ = torch.empty(1, device="cuda:0")

    _log("alloc_vmm_buffer")
    alloc = alloc_vmm_buffer(4 * 1024 * 1024, device=0)
    _log(f"allocated: va=0x{alloc.va:x} size={alloc.size}")

    _log("wrap_as_torch_uint8")
    buf = wrap_as_torch_uint8(alloc)
    _log(f"wrapped: shape={buf.shape} ptr=0x{buf.data_ptr():x}")

    # Simulate model_runner: create views in a loop, copy out
    named_tensors = []
    chunk = buf.size(0) // 4
    for i in range(4):
        t = buf[i * chunk : (i + 1) * chunk].view(torch.float32)
        named_tensors.append((f"layer.{i}.weight", t))

    _log(f"created {len(named_tensors)} views")

    # Simulate load_weights: copy_() from each view into a target param.
    # Use index loop to avoid creating a stale `view` loop variable —
    # model_runner only has one loop so there's no leaked iterator var.
    for idx in range(len(named_tensors)):
        tgt = torch.empty_like(named_tensors[idx][1])
        tgt.copy_(named_tensors[idx][1])
    torch.cuda.synchronize()
    _log("load_weights simulation done")

    # Check refcounts before cleanup
    _log(f"buf refcount = {sys.getrefcount(buf)}")
    _log(f"named_tensors refcount = {sys.getrefcount(named_tensors)}")
    _log(f"t refcount = {sys.getrefcount(t)}")

    # Exact model_runner cleanup — left-to-right del
    _log("del named_tensors, t, buf")
    del named_tensors, t, buf
    _log("free_vmm_buffer")
    free_vmm_buffer(alloc, close_fd=True)
    _log("DONE")


def _scenario_base_deleted_before_views():
    """
    BUG REPRODUCER: delete base tensor while views still exist.
    With refs on alloc (current approach), alloc outlives everything → safe.
    """
    import gc

    import torch

    _log("importing vmm_ipc")
    from sglang.srt.weight_sync.vmm_ipc import (
        alloc_vmm_buffer,
        free_vmm_buffer,
        wrap_as_torch_uint8,
    )

    _log("CUDA init")
    torch.cuda.init()
    torch.cuda.set_device(0)
    _ = torch.empty(1, device="cuda:0")

    _log("alloc + wrap")
    alloc = alloc_vmm_buffer(4 * 1024 * 1024, device=0)
    buf = wrap_as_torch_uint8(alloc)

    view_a = buf[: buf.size(0) // 2].view(torch.float32)
    view_b = buf[buf.size(0) // 2 :].view(torch.float32)
    _log("created 2 views")

    tgt = torch.empty_like(view_a)
    tgt.copy_(view_a)
    torch.cuda.synchronize()

    _log("del buf (views still alive)")
    del buf
    gc.collect()

    _log("free_vmm_buffer")
    free_vmm_buffer(alloc, close_fd=True)

    _log("del views")
    del view_a, view_b
    gc.collect()
    _log("DONE")


def _scenario_free_before_del():
    """
    BUG REPRODUCER (old code): free_vmm_buffer before tensor is deleted.
    Old code: _dlpack_refs = None → freed callback → segfault.
    Fixed code: _dlpack_refs NOT cleared → callback survives → safe.
    """
    import gc

    import torch

    _log("importing vmm_ipc")
    from sglang.srt.weight_sync.vmm_ipc import (
        alloc_vmm_buffer,
        free_vmm_buffer,
        wrap_as_torch_uint8,
    )

    _log("CUDA init")
    torch.cuda.init()
    torch.cuda.set_device(0)
    _ = torch.empty(1, device="cuda:0")

    _log("alloc + wrap")
    alloc = alloc_vmm_buffer(2 * 1024 * 1024, device=0)
    buf = wrap_as_torch_uint8(alloc)
    torch.cuda.synchronize()

    _log("free_vmm_buffer (tensor still alive)")
    free_vmm_buffer(alloc, close_fd=True)

    _log("del buf")
    del buf
    gc.collect()
    _log("DONE")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _HAS_CUDA, reason="CUDA device required")
def test_model_runner_cleanup_pattern():
    """Model-runner pattern: del named_tensors, t, buf then free_vmm_buffer."""
    code, err = _run_in_subprocess(_scenario_model_runner_pattern)
    assert code == 0, f"Subprocess failed (exitcode={code}):\n{err}"


@pytest.mark.skipif(not _HAS_CUDA, reason="CUDA device required")
def test_base_deleted_before_views():
    """Base tensor deleted while views survive.  Safe as long as alloc
    (which holds _dlpack_refs) outlives everything."""
    code, err = _run_in_subprocess(_scenario_base_deleted_before_views)
    assert code == 0, f"Subprocess crashed (exitcode={code}):\n{err}"


@pytest.mark.skipif(not _HAS_CUDA, reason="CUDA device required")
def test_free_vmm_before_tensor_del():
    """free_vmm_buffer before tensor del.  Safe as long as _dlpack_refs
    is NOT cleared by free_vmm_buffer."""
    code, err = _run_in_subprocess(_scenario_free_before_del)
    assert code == 0, f"Subprocess crashed (exitcode={code}):\n{err}"
