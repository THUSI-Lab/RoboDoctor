"""Select the project-local NVRTC before Isaac/torch load CUDA libraries."""
import ctypes
from pathlib import Path

_nvrtc_library = None


def preload_nvrtc():
    """Keep the shared Python environment unchanged; only affect this process."""
    global _nvrtc_library
    library = (Path(__file__).resolve().parent.parent /
               'runtime/curobo_nvrtc/nvidia/cuda_nvrtc/lib/libnvrtc.so.12')
    if library.is_file():
        _nvrtc_library = ctypes.CDLL(str(library), mode=ctypes.RTLD_GLOBAL)
        return str(library)
    return None


def nvrtc_info():
    from cuda.bindings import nvrtc
    status, major, minor = nvrtc.nvrtcVersion()
    if int(status) != 0:
        raise RuntimeError(f'Cannot query NVRTC version: {status}')
    status, archs = nvrtc.nvrtcGetSupportedArchs()
    if int(status) != 0:
        raise RuntimeError(f'Cannot query NVRTC architectures: {status}')
    return dict(nvrtc_version=f'{major}.{minor}', supported_architectures=list(archs))


def check_nvrtc_architecture(major, minor):
    info = nvrtc_info()
    arch = major * 10 + minor
    if arch not in info['supported_architectures']:
        raise RuntimeError(
            f'cuRobo requires sm_{arch}, but loaded NVRTC {info["nvrtc_version"]} '
            f'does not support it. Install the local compiler with '
            f'./setup_curobo_runtime.sh and restart the simulator.')
    return dict(info, gpu_architecture=f'sm_{arch}')
