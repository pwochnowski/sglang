"""
Two-process round-trip test for vmm_ipc.py.

Uses multiprocessing spawn (CUDA contexts don't survive fork).
Run inside the verl docker container with a CUDA device available:
    pytest example/verl/tests/test_vmm_ipc.py -v -s
"""

import gc
import multiprocessing
import os
import struct
import sys

import pytest

# # Ensure the verl source tree is importable (tests/ sits outside the package).
# _VERL_SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "verl")

try:
    import torch

    _HAS_CUDA = torch.cuda.is_available()
except ImportError:
    _HAS_CUDA = False

_TIMEOUT = 30  # seconds — per blocking operation


# ---------------------------------------------------------------------------
# Subprocess entry points (module-level so they are picklable by spawn)
# ---------------------------------------------------------------------------


def _log(tag: str, msg: str):
    sys.stderr.write(f"[{tag}] {msg}\n")
    sys.stderr.flush()


def _sender_proc(uds_path: str, pattern_seed: int, listener_ready, result_queue):
    """Sender: allocate VMM buffer, fill, export fd, wait for ack, free."""
    try:
        # sys.path.insert(0, _VERL_SRC)
        _log("sender", "importing vmm_ipc")
        import torch
        import vmm_ipc

        _log("sender", "CUDA init")
        torch.cuda.init()
        torch.cuda.set_device(0)
        _ = torch.empty(1, device="cuda:0")  # ensure CUDA context exists

        _log("sender", "alloc_vmm_buffer(1 MiB)")
        alloc = vmm_ipc.alloc_vmm_buffer(1 << 20, device=0)
        _log("sender", f"allocated: va=0x{alloc.va:x} size={alloc.size} fd={alloc.fd}")

        t = vmm_ipc.wrap_as_torch_uint8(alloc)
        _log("sender", f"wrapped: shape={t.shape} ptr=0x{t.data_ptr():x}")

        # Fill with deterministic pattern
        torch.manual_seed(pattern_seed)
        pattern = torch.randint(
            0, 256, (alloc.size,), dtype=torch.uint8, device="cuda:0"
        )
        t.copy_(pattern)
        torch.cuda.synchronize()
        _log("sender", "buffer filled")

        # Bind listener, signal readiness, then accept
        listener = vmm_ipc.open_sidecar_listener(uds_path)
        listener.settimeout(_TIMEOUT)
        _log("sender", f"listener bound at {uds_path}, signalling ready")
        listener_ready.set()
        conn, _ = listener.accept()
        conn.settimeout(_TIMEOUT)
        _log("sender", "receiver connected")

        # Send the rounded size then the fd
        conn.send(struct.pack("Q", alloc.size))
        vmm_ipc.send_fd(conn, alloc.fd)
        _log("sender", "sent size + fd")

        # Wait for receiver's ack (means it has verified the data)
        ack = conn.recv(1)
        assert ack == b"\x00", f"Expected ack byte, got {ack!r}"
        _log("sender", "received ack")

        # Teardown: tensor first, then VMM, then sockets
        conn.close()
        listener.close()
        try:
            os.unlink(uds_path)
        except FileNotFoundError:
            pass

        del t
        gc.collect()
        vmm_ipc.free_vmm_buffer(alloc, close_fd=True)
        _log("sender", "cleanup done")

        result_queue.put(("ok", None))
    except Exception:
        import traceback

        _log("sender", f"FAILED: {traceback.format_exc()}")
        result_queue.put(("error", traceback.format_exc()))
    finally:
        # Flush the queue's background feeder thread, then skip interpreter
        # shutdown to avoid ctypes/CUDA teardown-order segfaults.
        result_queue.close()
        result_queue.join_thread()
        os._exit(0)


def _receiver_proc(uds_path: str, pattern_seed: int, listener_ready, result_queue):
    """Receiver: connect, receive fd, import buffer, verify data, ack, free."""
    try:
        # sys.path.insert(0, _VERL_SRC)
        _log("receiver", "importing vmm_ipc")
        import torch
        from import vmm_ipc

        _log("receiver", "CUDA init")
        torch.cuda.init()
        torch.cuda.set_device(0)
        _ = torch.empty(1, device="cuda:0")

        # Wait until sender has bound the listener
        _log("receiver", "waiting for listener_ready event")
        assert listener_ready.wait(timeout=_TIMEOUT), "Timed out waiting for sender"

        _log("receiver", "connecting to sender")
        client = vmm_ipc.open_sidecar_client(uds_path)
        client.settimeout(_TIMEOUT)
        _log("receiver", "connected")

        # Receive rounded size, then fd
        size_data = client.recv(8)
        size = struct.unpack("Q", size_data)[0]
        fd = vmm_ipc.recv_fd(client)
        _log("receiver", f"received size={size} fd={fd}")

        alloc = vmm_ipc.import_vmm_buffer(fd, size, device=0)
        _log("receiver", f"imported: va=0x{alloc.va:x} size={alloc.size}")

        t = vmm_ipc.wrap_as_torch_uint8(alloc)
        _log("receiver", f"wrapped: shape={t.shape} ptr=0x{t.data_ptr():x}")

        # Verify the pattern matches sender byte-for-byte
        torch.manual_seed(pattern_seed)
        expected = torch.randint(
            0, 256, (size,), dtype=torch.uint8, device="cuda:0"
        )
        torch.cuda.synchronize()
        match = torch.equal(t, expected)
        if not match:
            got = t[:16].cpu().tolist()
            want = expected[:16].cpu().tolist()
            n_diff = int((t != expected).sum().item())
            _log("receiver", f"MISMATCH: {n_diff}/{size} bytes differ")
            _log("receiver", f"  got[:16]  = {got}")
            _log("receiver", f"  want[:16] = {want}")
        _log("receiver", f"data verification: {'PASS' if match else 'FAIL'}")
        assert match, "Data mismatch: imported buffer != expected"

        # Ack to sender so it can safely free
        client.send(b"\x00")
        client.close()
        _log("receiver", "ack sent")

        del t
        gc.collect()
        vmm_ipc.free_vmm_buffer(alloc, close_fd=True)
        _log("receiver", "cleanup done")

        result_queue.put(("ok", None))
    except Exception:
        import traceback

        _log("receiver", f"FAILED: {traceback.format_exc()}")
        result_queue.put(("error", traceback.format_exc()))
    finally:
        # Flush the queue's background feeder thread, then skip interpreter
        # shutdown to avoid ctypes/CUDA teardown-order segfaults.
        result_queue.close()
        result_queue.join_thread()
        os._exit(0)


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _HAS_CUDA, reason="CUDA device required")
def test_vmm_ipc_roundtrip():
    """Allocate VMM buffer in one process, export fd, import in another, verify."""
    ctx = multiprocessing.get_context("spawn")

    uds_path = f"/tmp/test-vmm-ipc-{os.getpid()}.sock"
    pattern_seed = 42

    listener_ready = ctx.Event()
    sender_q = ctx.Queue()
    receiver_q = ctx.Queue()

    sender = ctx.Process(
        target=_sender_proc, args=(uds_path, pattern_seed, listener_ready, sender_q)
    )
    receiver = ctx.Process(
        target=_receiver_proc,
        args=(uds_path, pattern_seed, listener_ready, receiver_q),
    )

    sender.start()
    receiver.start()

    try:
        sender.join(timeout=90)
        receiver.join(timeout=90)
    finally:
        # Don't leave hung children
        for p in (sender, receiver):
            if p.is_alive():
                p.terminate()
                p.join(timeout=5)

    # Collect errors from both sides before asserting, so we see both on failure
    import queue
    try:
        s_status, s_err = sender_q.get(timeout=5)
    except queue.Empty:
        print("Empty sender queue")
        s_status, s_err = None, None
    try:
        r_status, r_err = receiver_q.get(timeout=5)
    except queue.Empty:
        print("Empty recv queue")
        r_status, r_err = None, None

    failures = []
    if s_status is None:
        failures.append(f"Sender: no result (exitcode={sender.exitcode})")
    elif s_status != "ok":
        failures.append(f"Sender:\n{s_err}")
    if r_status is None:
        failures.append(f"Receiver: no result (exitcode={receiver.exitcode})")
    elif r_status != "ok":
        failures.append(f"Receiver:\n{r_err}")

    assert not failures, "\n\n".join(failures)