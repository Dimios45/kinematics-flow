#!/bin/bash
# Full multi-embodiment training on a single MI300X (paper config: 25000 scenes,
# 5 grippers + z0, batch 5 scenes/step, 500 epochs, checkpoint every 5 epochs).
# Usage: ./launch_train.sh [extra hydra overrides...]
cd "$(dirname "$0")" || exit 1

SP="$PWD/.venv/lib/python3.11/site-packages"
export LD_LIBRARY_PATH="$PWD/.rocm-shim:$SP/_rocm_sdk_core/lib:$SP/_rocm_sdk_core/lib/rocm_sysdeps/lib:$SP/_rocm_sdk_libraries_gfx94X_dcgpu/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export MUJOCO_GL=egl
# cache with the verified GPU-2 autotune results; the old ~/.jax_cache holds
# executables whose baked-in rocBLAS solution choices crash (see TRAINING_DIARY.md)
export JAX_COMPILATION_CACHE_DIR=${JAX_COMPILATION_CACHE_DIR:-/mnt/data/mritunjoyh/.jax_cache_gpu2}

# Defaults = the configuration verified on 2026-07-08 (TRAINING_DIARY.md, Diagnostic E):
# GPU 2 is the only GPU where this workload runs next to its co-tenants — GPU 1
# crashes (hipBLASLt on) or deadlocks (hipBLASLt off) in rocBLAS.
export HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-2}
# On-demand allocation, exactly as verified. Preallocating a tight pool (.44) starved
# rocBLAS/autotune of raw VRAM and reintroduced the internal error (see diary, 12:03).
# Measured true footprint: ~86 GB peak next to a ~109 GB co-tenant.
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=${XLA_PYTHON_CLIENT_MEM_FRACTION:-.95}
export ROCBLAS_USE_HIPBLASLT=${ROCBLAS_USE_HIPBLASLT:-0}
# Repeated deadlocks in kfd_wait_on_events (a GPU event that never signals) at varying
# program points while inference tenants hammer the same GPU -> avoid the shared SDMA
# copy engines; copies go through compute (blit) kernels instead.
export HSA_ENABLE_SDMA=0
# Limit our HW queue footprint next to inference tenants (queue oversubscription is
# another documented cause of KFD event hangs on shared MI300X).
export GPU_MAX_HW_QUEUES=2
# rocBLAS lazy first-use init inside HIP graph capture fails with
# rocblas_status_internal_error on this stack -> disable XLA command buffers.
# autotune_level=0: GEMM autotune probing/solution replay caused crashes and GPU
# deadlocks next to busy co-tenants (diary 2026-07-08); use default algorithms.
# triton_gemm=false: with autotune off, XLA's heuristic routed a dot to the Triton
# emitter which rejects it ("Contracting dimension is too fragmented") — force all
# dots down the plain rocBLAS default-algorithm path instead.
export XLA_FLAGS="--xla_gpu_enable_command_buffer= --xla_gpu_autotune_level=0 --xla_gpu_enable_triton_gemm=false ${XLA_FLAGS:-}"

# -u: unbuffered stdout. With stdout redirected to a log file, buffered "Train Step"
# prints stay invisible for hours and healthy runs look deadlocked (diary 2026-07-08).
exec .venv/bin/python -u -m kin_flow.cli.train save_every_epoch=5 "$@"
