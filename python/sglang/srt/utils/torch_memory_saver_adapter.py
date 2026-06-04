import logging
from abc import ABC
from contextlib import contextmanager, nullcontext

from sglang.srt.constants import GPU_MEMORY_TYPE_KV_CACHE, GPU_MEMORY_TYPE_WEIGHTS

try:
    import torch_memory_saver

    _memory_saver = torch_memory_saver.torch_memory_saver
    import_error = None
except ImportError as e:
    import_error = e
    pass

try:
    import gcr
    _has_gcr = True
except ImportError:
    _has_gcr = False

_GCR_TAG_MAP = {
    GPU_MEMORY_TYPE_KV_CACHE: 1,
    GPU_MEMORY_TYPE_WEIGHTS: 2,
}

_GCR_EPHEMERAL_TYPES = {GPU_MEMORY_TYPE_KV_CACHE}
_GCR_TAGGED_TYPES = {GPU_MEMORY_TYPE_KV_CACHE, GPU_MEMORY_TYPE_WEIGHTS}


@contextmanager
def _gcr_context(tag: str):
    if not _has_gcr or tag not in _GCR_TAG_MAP:
        yield
        return
    gcr_tag = _GCR_TAG_MAP[tag]
    with gcr.ephemeral() if tag in _GCR_EPHEMERAL_TYPES else nullcontext():
        with gcr.tagged(tag=gcr_tag) if tag in _GCR_TAGGED_TYPES else nullcontext():
            yield

logger = logging.getLogger(__name__)


class TorchMemorySaverAdapter(ABC):
    @staticmethod
    def create(enable: bool):
        if enable and import_error is not None:
            logger.warning(
                "enable_memory_saver is enabled, but "
                "torch-memory-saver is not installed. Please install it "
                "via `pip3 install torch-memory-saver`. "
            )
            raise import_error
        return (
            _TorchMemorySaverAdapterReal() if enable else _TorchMemorySaverAdapterNoop()
        )

    def check_validity(self, caller_name):
        if not self.enabled:
            logger.warning(
                f"`{caller_name}` will not save memory because torch_memory_saver is not enabled. "
                f"Potential causes: `enable_memory_saver` is false, or torch_memory_saver has installation issues."
            )

    def configure_subprocess(self):
        raise NotImplementedError

    def region(self, tag: str, enable_cpu_backup: bool = False):
        raise NotImplementedError

    def cuda_graph(self, **kwargs):
        raise NotImplementedError

    def disable(self):
        raise NotImplementedError

    def pause(self, tag: str):
        raise NotImplementedError

    def resume(self, tag: str):
        raise NotImplementedError

    @property
    def enabled(self):
        raise NotImplementedError


class _TorchMemorySaverAdapterReal(TorchMemorySaverAdapter):
    """Adapter for TorchMemorySaver with tag-based control"""

    def configure_subprocess(self):
        return torch_memory_saver.configure_subprocess()

    @contextmanager
    def region(self, tag: str, enable_cpu_backup: bool = False):
        with _memory_saver.region(tag=tag, enable_cpu_backup=enable_cpu_backup):
            if tag in _GCR_TAG_MAP:
                with _gcr_context(tag=tag):
                    yield
            else:
                yield

    def cuda_graph(self, **kwargs):
        return _memory_saver.cuda_graph(**kwargs)

    def disable(self):
        return _memory_saver.disable()

    def pause(self, tag: str):
        return _memory_saver.pause(tag=tag)

    def resume(self, tag: str):
        return _memory_saver.resume(tag=tag)

    @property
    def enabled(self):
        return _memory_saver is not None and _memory_saver.enabled


class _TorchMemorySaverAdapterNoop(TorchMemorySaverAdapter):
    @contextmanager
    def configure_subprocess(self):
        yield

    @contextmanager
    def region(self, tag: str, enable_cpu_backup: bool = False):
        if tag in _GCR_TAG_MAP:
            with _gcr_context(tag=tag):
                yield
        else:
            yield

    @contextmanager
    def cuda_graph(self, **kwargs):
        yield

    @contextmanager
    def disable(self):
        yield

    def pause(self, tag: str):
        pass

    def resume(self, tag: str):
        pass

    @property
    def enabled(self):
        return False
