#!/bin/bash
# Full 5-gripper eval of the final epoch-120 model (me-full_25000_resume_step302000).
# Same proven ROCm env as scripts/launch_bench.sh (which now lives in scripts/ and
# whose internal cd breaks); GPU 2, sequential grippers, one log per gripper.
cd /mnt/data/mritunjoyh/kinematics-flow || exit 1

SP="$PWD/.venv/lib/python3.11/site-packages"
export LD_LIBRARY_PATH="$PWD/.rocm-shim:$SP/_rocm_sdk_core/lib:$SP/_rocm_sdk_core/lib/rocm_sysdeps/lib:$SP/_rocm_sdk_libraries_gfx94X_dcgpu/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
unset MUJOCO_GL
export MGS_NO_RENDER=1
export JAX_COMPILATION_CACHE_DIR=/mnt/data/mritunjoyh/.jax_cache_gpu2
export HIP_VISIBLE_DEVICES=2
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=.90
export ROCBLAS_USE_HIPBLASLT=0
export HSA_ENABLE_SDMA=0
export GPU_MAX_HW_QUEUES=2
export XLA_FLAGS="--xla_gpu_enable_command_buffer= --xla_gpu_autotune_level=0 --xla_gpu_enable_triton_gemm=false"

CKPT="me-full_25000_resume_step302000"
for g in 0 1 2 3 4; do
  echo "=== gripper_id=$g start $(date '+%F %T') ==="
  .venv/bin/python -u -m kin_flow.cli.bench \
    checkpoint="$CKPT" gripper_id=$g num_scenes=10 \
    > "bench_ep120true_gripper$g.log" 2>&1
  echo "=== gripper_id=$g exit=$? end $(date '+%F %T') ==="
done
echo "ALL DONE $(date '+%F %T')"
