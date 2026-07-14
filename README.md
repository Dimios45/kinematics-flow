# Kinematics Flow

Repository provides training pipeline and dataset for the method presented in [Towards a Multi-Embodied Grasping Agent](https://arxiv.org/abs/2510.27420).

### Installation and Setup

The project assumes the following structure.

```
root/
├── data/
│   ├── train/          <- training scenes for all five grippers
│   └── test/           <- testing scenes for all five grippers
├── kinematics-flow/    <- this repository
└── checkpoint/         <- storage for model weights
```

In case you use [mise-en-place](https://mise.jdx.dev/) install via:
```bash
mise i && uv sync
```
Project will automatically source .venv directory and set correct globals.
Without mise you can use just `uv` for python 3.11.13 (tested version):
```bash
uv sync
```
Then manually set the following environment variables.

```bash
CONFIG_ROOT="<YOUR PROJECT ROOT>"
CUDA_LIBS="${CONFIG_ROOT}/.venv/lib/python3.11/site-packages/nvidia/cublas/lib"
CUDA_LIBS="${CUDA_LIBS}:${CONFIG_ROOT}/.venv/lib/python3.11/site-packages/nvidia/cuda_runtime/lib"
CUDA_LIBS="${CUDA_LIBS}:${CONFIG_ROOT}/.venv/lib/python3.11/site-packages/nvidia/cuda_nvrtc/lib"
CUDA_LIBS="${CUDA_LIBS}:${CONFIG_ROOT}/.venv/lib/python3.11/site-packages/nvidia/nvjitlink/lib"
export LD_LIBRARY_PATH="${CUDA_LIBS}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export MUJOCO_GL="egl"
```

### AMD ROCm (MI300X) setup

This checkout has been adapted to run on AMD Instinct GPUs (tested: 8× MI300X, gfx942,
host ROCm 6.2.4 kernel driver, no sudo needed). `pyproject.toml` was changed to
`jax==0.8.0`, `flax==0.11.2`, and the NVIDIA-only `cuequivariance-ops-jax-cu12` was
dropped (the model only uses the pure-XLA `method="naive"` path). Install with:

```bash
uv venv --python 3.11
uv pip install --prerelease=allow \
  --extra-index-url "https://rocm.nightlies.amd.com/v2/gfx94X-dcgpu/" \
  --index-strategy unsafe-best-match \
  "jax==0.8.0" \
  "jax-rocm7-plugin==0.8.0+rocm7.13.0a20260421" \
  "jax-rocm7-pjrt==0.8.0+rocm7.13.0a20260421" \
  "rocm[libraries]==7.13.0a20260421"
uv pip install -e ./thrd_party/mj-grasp-sim -e .
```

The pip-installed ROCm SDK is missing two `rocm_sysdeps` libraries that the JAX plugin
links against; `.rocm-shim/` contains symlinks to the system `libhwloc`/`libpciaccess`
as stand-ins (recreate with `ln -sf /usr/lib/x86_64-linux-gnu/libhwloc.so.15
.rocm-shim/librocm_sysdeps_hwloc.so.5` etc. if needed). `mise.toml` sets
`LD_LIBRARY_PATH` accordingly; without mise export:

```bash
SP="$PWD/.venv/lib/python3.11/site-packages"
export LD_LIBRARY_PATH="$PWD/.rocm-shim:$SP/_rocm_sdk_core/lib:$SP/_rocm_sdk_core/lib/rocm_sysdeps/lib:$SP/_rocm_sdk_libraries_gfx94X_dcgpu/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export MUJOCO_GL="egl"
export JAX_COMPILATION_CACHE_DIR="$HOME/.jax_cache"   # first XLA compile takes ~5 min
```

Do not run `uv sync` afterwards — it would remove the ROCm plugin packages, which are
installed imperatively (they are not in `pyproject.toml`).

ROCm-specific notes:

- **Train on a single GPU.** The full multi-embodiment model fits in one MI300X
  (192 GB; verified ~2 s/step). Select a free GPU with e.g. `HIP_VISIBLE_DEVICES=2`
  and set `XLA_PYTHON_CLIENT_MEM_FRACTION` to fit alongside co-tenant jobs.
- **Multi-GPU `pmap` in one process is currently broken** on this stack:
  concurrent rocBLAS use across device threads fails with
  `rocblas_status_internal_error` (a sequential warmup in `train.py` moves but
  does not eliminate the failure). Revisit after a ROCm/rocBLAS update.
- On GPUs busy with other tenants, single-GPU runs may also hit the rocBLAS error;
  `export ROCBLAS_USE_HIPBLASLT=1` worked around it in testing.
- flax was bumped to 0.11.2 (jax 0.8 compatibility); `nnx.ModelAndOptimizer`
  replaces `nnx.Optimizer` in `kin_flow/ctrl/trainer.py`, and
  `TPWithWeightsAndBiases` (`kin_flow/net/module/fctp.py`) stores per-path
  `nnx.Param`s — checkpoints written with the old Param-of-list layout would
  need remapping.

For the simulator download the object assets as described in https://github.com/boschresearch/mj-grasp-sim
and place them under the directory `./thrd_party/mj-grasp-sim/asset/mj-objects/`. Make sure
to also place the `fast_eta_objects.txt` in the same directory to select the proper
object subset for faster simulation performance.

Download the dataset from (RELEASE SOON) and place the train and test directories in the root data
directory as described above.

### Training and Testing

Training the full model requires over 100GB VRAM (tested on H200) and can be initiated via
```bash
python -m kin_flow.cli.train save_every_epoch=5
```
Expect reasonable convergence after 120 epochs (roughly 1 week of training).

To train single-embodiment configurations choose one of the following.
Note training is always bottlenecked by highest DoF gripper due to zero-padding for mixed models.

```bash
python -m kin_flow.cli.train gripper=["allegro"] globals.MAX_DOF=16 run_name="se-allegro" num_scenes=5000 num_scenes_per_batch=5 epochs=500 save_every_epoch=50
python -m kin_flow.cli.train gripper=["dexee"] globals.MAX_DOF=12 run_name="se-dexee" num_scenes=5000 num_scenes_per_batch=5 epochs=500 save_every_epoch=50
python -m kin_flow.cli.train gripper=["panda"] globals.MAX_DOF=2 run_name="se-panda" num_scenes=5000 num_scenes_per_batch=5 epochs=500 save_every_epoch=50
python -m kin_flow.cli.train gripper=["shadow"] globals.MAX_DOF=22 run_name="se-shadow" num_scenes=5000 num_scenes_per_batch=5 epochs=500 save_every_epoch=50
python -m kin_flow.cli.train gripper=["vx300"] globals.MAX_DOF=2 run_name="se-vx" num_scenes=5000 num_scenes_per_batch=5 epochs=500 save_every_epoch=50
```
Configurations are managed via Hydra, see `./kin_flow/cli/config/` for all command line configurable
model settings.

Test your trained checkpoint via
```bash
python -m kin_flow.cli.bench checkpoint="<your checkpoint>" gripper_id=<id from the gripper list>
```
The gripper list is ordered as: `0=panda, 1=vx300, 2=dexee, 3=allegro, 4=shadow`. Note for single single-embodiment
models you should set `z0_id` to -1 (disabled)

To visualize the grasps use
```bash
python -m kin_flow.cli.bench checkpoint="<your checkpoint>" gripper_id=<id from the gripper list> num_samples=10 num_scenes=1 viz=true gs=false
```

### Limitations

1. While a working PoC. For sim2real, we only validated this setup on the Panda Hardware platform.
Settings were optimized to generate sufficient large-scale data quantity to train
the model. Note for the Allegro gripper we used position gain of 1.0 during
generation and 1.5 during testing as we later noticed that the gripper did drop objects
due to insufficient force applied to keep the gripper closed.

2. For faster evaluation we use the simulation collision check to filter out
surface penetrations by the grippers. For analytical filtering we have tested
using the collision models itself on the raw point cloud to identify collisions,
which nearly coincides with the ground truth simulation collision predictions
and allows real world deployment. However, this code is not part of this work.

3. Zero-shot grasp generation for other gripper types has not been tested and
would at a minimum require finetuning of the gripper embeddings.

### License

Kinematics Flow and MuJoCo Grasping Simulator is open-sourced under the AGPL-3.0 license. See the
[LICENSE](LICENSE) file for details.

For a list of other open source components included in Kinematics Flow or MuJoCo Grasp Simulator, see the
file [3rd-party-licenses.txt](3rd-party-licenses.txt).
