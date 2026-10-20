import pytest

from robodoctor import cuda_runtime


def test_rejects_sm103_with_nvrtc128_before_planner_warmup(monkeypatch):
    monkeypatch.setattr(cuda_runtime, 'nvrtc_info', lambda: dict(
        nvrtc_version='12.8', supported_architectures=[80, 90, 100, 101, 120]))
    with pytest.raises(RuntimeError, match=r'sm_103.*NVRTC 12.8.*setup_curobo_runtime.sh'):
        cuda_runtime.check_nvrtc_architecture(10, 3)


def test_accepts_exact_gpu_architecture_without_downgrading(monkeypatch):
    monkeypatch.setattr(cuda_runtime, 'nvrtc_info', lambda: dict(
        nvrtc_version='12.9', supported_architectures=[90, 100, 103, 120]))
    result = cuda_runtime.check_nvrtc_architecture(10, 3)
    assert result['gpu_architecture'] == 'sm_103'
    assert result['nvrtc_version'] == '12.9'


def test_preloads_project_library_globally_without_importing_isaac(monkeypatch, tmp_path):
    monkeypatch.setattr(cuda_runtime, '__file__', str(tmp_path/'robodoctor/cuda_runtime.py'))
    library = tmp_path/'runtime/curobo_nvrtc/nvidia/cuda_nvrtc/lib/libnvrtc.so.12'
    library.parent.mkdir(parents=True)
    library.write_bytes(b'fake shared library')
    calls = []
    monkeypatch.setattr(cuda_runtime.ctypes, 'CDLL', lambda path, mode: calls.append((path, mode)))
    assert cuda_runtime.preload_nvrtc() == str(library)
    assert calls == [(str(library), cuda_runtime.ctypes.RTLD_GLOBAL)]


def test_no_local_library_uses_existing_runtime(monkeypatch, tmp_path):
    monkeypatch.setattr(cuda_runtime, '__file__', str(tmp_path/'robodoctor/cuda_runtime.py'))
    assert cuda_runtime.preload_nvrtc() is None
