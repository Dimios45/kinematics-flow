#!/bin/bash
# Eval/bench wrapper: same proven ROCm env as launch_train.sh, but defaults to
# GPU 2 (training occupies GPU 3; GPU 1 is broken for this workload — diary).
# Usage: ./launch_bench.sh checkpoint="me-full_25000_5" gripper_id=0 [overrides...]
cd "$(dirname "$0")" || exit 1

SP="$PWD/.venv/lib/python3.11/site-packages"
export LD_LIBRARY_PATH="$PWD/.rocm-shim:$SP/_rocm_sdk_core/lib:$SP/_rocm_sdk_core/lib/rocm_sysdeps/lib:$SP/_rocm_sdk_libraries_gfx94X_dcgpu/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
# do NOT set MUJOCO_GL: egl segfaults `import mujoco` on this node (headless EGL vs
# ROCm driver), and bench is physics-only — no rendering context needed.
unset MUJOCO_GL
export MGS_NO_RENDER=1   # skip mujoco.Renderer creation (no GL backend works here)
export JAX_COMPILATION_CACHE_DIR=${JAX_COMPILATION_CACHE_DIR:-/mnt/data/mritunjoyh/.jax_cache_gpu2}

export HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-2}
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=.90
export ROCBLAS_USE_HIPBLASLT=0
export HSA_ENABLE_SDMA=0
export GPU_MAX_HW_QUEUES=2
export XLA_FLAGS="--xla_gpu_enable_command_buffer= --xla_gpu_autotune_level=0 --xla_gpu_enable_triton_gemm=false ${XLA_FLAGS:-}"

exec .venv/bin/python -u -m kin_flow.cli.bench "$@"
