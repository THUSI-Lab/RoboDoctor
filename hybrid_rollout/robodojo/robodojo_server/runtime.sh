#!/usr/bin/env bash
# Source this file in the RoboDojo process; it does not modify system libraries.
export TASK_ROOT="${TASK_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)/runtime}"
DAGGER_DRIVER_VERSION="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -n 1)"
if [[ -z "${DAGGER_NVIDIA_RUNTIME:-}" ]]; then
  # System libraries are discovered; a custom bundle must be configured explicitly.
  for graphics_dir in /usr/lib/x86_64-linux-gnu /usr/lib64 /usr/local/nvidia/lib64; do
    if [[ -f "$graphics_dir/libGLX_nvidia.so.$DAGGER_DRIVER_VERSION" ]]; then
      export DAGGER_NVIDIA_RUNTIME="$graphics_dir"
      if [[ "$graphics_dir" == /usr/* ]]; then
        export DAGGER_SYSROOT="${DAGGER_SYSROOT:-/}"
      fi
      break
    fi
  done
fi
export DAGGER_NVIDIA_RUNTIME="${DAGGER_NVIDIA_RUNTIME:-}"
export DAGGER_SYSROOT="${DAGGER_SYSROOT:-/}"
if [[ ! -f "$DAGGER_NVIDIA_RUNTIME/libGLX_nvidia.so.$DAGGER_DRIVER_VERSION" ]]; then
  printf 'Missing NVIDIA userspace graphics libraries matching driver %s in %s\n' "$DAGGER_DRIVER_VERSION" "$DAGGER_NVIDIA_RUNTIME" >&2
  return 1 2>/dev/null || exit 1
fi
export LD_LIBRARY_PATH="$DAGGER_NVIDIA_RUNTIME:${DAGGER_SYSROOT%/}/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
if [[ -z "${VK_ICD_FILENAMES:-}" ]]; then
  for graphics_manifest in "$DAGGER_NVIDIA_RUNTIME/nvidia_icd.json" \
      /etc/vulkan/icd.d/nvidia_icd.json /usr/share/vulkan/icd.d/nvidia_icd.json; do
    if [[ -f "$graphics_manifest" ]]; then
      export VK_ICD_FILENAMES="$graphics_manifest"
      break
    fi
  done
fi
if [[ -z "${__EGL_VENDOR_LIBRARY_FILENAMES:-}" ]]; then
  for graphics_manifest in "$DAGGER_NVIDIA_RUNTIME/10_nvidia.json" \
      /usr/share/glvnd/egl_vendor.d/10_nvidia.json /etc/glvnd/egl_vendor.d/10_nvidia.json; do
    if [[ -f "$graphics_manifest" ]]; then
      export __EGL_VENDOR_LIBRARY_FILENAMES="$graphics_manifest"
      break
    fi
  done
fi
if [[ -z "${VK_ICD_FILENAMES:-}" || -z "${__EGL_VENDOR_LIBRARY_FILENAMES:-}" ]]; then
  echo 'Missing NVIDIA Vulkan/EGL manifests; set VK_ICD_FILENAMES and __EGL_VENDOR_LIBRARY_FILENAMES' >&2
  return 1 2>/dev/null || exit 1
fi
export XDG_RUNTIME_DIR="$TASK_ROOT/runtime/xdg"
mkdir -p "$XDG_RUNTIME_DIR"
chmod 700 "$XDG_RUNTIME_DIR"
export OMNI_KIT_ACCEPT_EULA=Y
