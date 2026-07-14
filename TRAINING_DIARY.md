# Training diary — me-full_25000 on MI300X

Chronological log of every launch attempt, crash, diagnosis, and fix for the full
multi-embodiment kinematics-flow training run. Maintained by Claude.

## 2026-07-07

- **20:44** Attempt 1 (user, GPU 1, mem fraction .80, ROCBLAS_USE_HIPBLASLT=1): crashed at
  first train step — `Failed to capture gpu graph: rocblas_gemm_strided_batched_ex …
  rocblas_status_internal_error`. Hypothesis: rocBLAS lazy init inside HIP graph capture.
  wandb run `au0bebi2` (crashed).
- **21:04** Attempt 2 (user, GPU 1, + `--xla_gpu_enable_command_buffer=` to disable graph
  capture): graph-capture prefix gone, but same rocBLAS internal error at first step.
  wandb run `pnhxshdq` (crashed). Hypothesis 2: rocBLAS device workspace starved
  (only ~4.6 GB VRAM free outside the XLA pool).
- **21:14** Attempt 3 (Claude, GPU 1, mem fraction .70 → ~25 GB free outside pool): same
  rocBLAS internal error ~90 s in. wandb run `sxeugse0` (crashed). Hypothesis 2 REJECTED —
  not a VRAM-headroom problem. Note: identical code + real-shape config previously trained
  fine on GPU 2 when it was free; GPU 1 has never passed with real-data shapes
  (15000-point clouds; smoke tests with 4096-point synthetic data DID pass on GPU 1 with
  ROCBLAS_USE_HIPBLASLT=1). Suspicion: shape-dependent rocBLAS kernel path broken/racing
  on GPU 1 alongside the co-tenant (36 GB resident, idle sglang/vLLM).
- **21:23** Diagnostic A (GPU 1, 50 scenes, wandb off): force Triton GEMMs
  (`--xla_gpu_enable_triton_gemm=true --xla_gpu_cublas_fallback=false`) — FAILED, same
  rocBLAS error. XLA still lowers these batched dots to rocBLAS on ROCm; the flag doesn't
  bypass it.
- **21:2x** KEY INSIGHT: all previous *successes* (incl. "GPU 2 works") used synthetic
  4096-point scenes. Real data (15000-point clouds) has never trained on ANY GPU here.
  The failure may be shape-dependent inside rocBLAS, not GPU-1-specific.
- **21:3x** Diagnostic B (GPU 1, 50 scenes): `ROCBLAS_LAYER=2` bench logging captured the
  failing call — a TINY gemm (m=640 n=64 k=5 fp32, batch_count=125, ldb=5) with
  `--algo 1 --solution_index -621285532`. Negative solution index = hipBLASLt-backed
  rocBLAS solution → prime suspect is `ROCBLAS_USE_HIPBLASLT=1` (added earlier as a
  busy-GPU workaround): XLA autotune selects a hipBLASLt solution that then fails to
  replay for this shape. NOT a big-shape/overflow problem, NOT VRAM headroom.
- **21:27** Diagnostic C (GPU 1, 50 scenes): `ROCBLAS_USE_HIPBLASLT=0` — INVALID TEST:
  launch_train.sh hard-exported JAX_COMPILATION_CACHE_DIR, so the old cache (with the
  poisoned solution_index baked into the executable) was reused; failed in seconds with
  the identical index. Script fixed to make the cache dir overridable.
  Lesson: the persistent JAX compile cache stores the autotuned rocBLAS solution choice;
  env-var changes alone don't invalidate it.
- **21:33** Diagnostic D (GPU 1, 50 scenes): `ROCBLAS_USE_HIPBLASLT=0` + genuinely fresh
  cache. Compiled fine (3m09s), autotune probes ran (98k logged rocBLAS calls, negative
  solution indices appear even with hipBLASLt off — they're normal in this rocBLAS build),
  then **HUNG at 100% CPU with no output from 21:44 onward; killed 14 h later**
  (2026-07-08 11:38). Last logged call: gemm_ex T/T m=256 n=256 k=2000 fp32.
  CONCLUSION: GPU 1 is unusable for this workload — hipBLASLt on → instant
  rocblas_status_internal_error on a k=5 strided-batched GEMM; hipBLASLt off → rocBLAS
  deadlock. Both failure modes are specific to GPU 1 + co-tenant.

## 2026-07-08

- **11:38** GPU survey: no fully-free GPU. Free VRAM: GPU3 ~181 GB, GPU0 ~178 GB,
  GPU1 ~170 GB (broken), GPU2 ~97 GB (the only GPU that ever ran this code
  successfully), GPU4–7 ~36–46 GB. All have busy co-tenant compute.
- **11:4x** Diagnostic E (GPU 2, 50 scenes): `XLA_PYTHON_CLIENT_PREALLOCATE=false` to
  measure the model's TRUE peak VRAM (never measured — earlier runs preallocated
  arbitrary fractions). hipBLASLt off (GPU 2's original success used defaults), fresh
  cache, 25-min watchdog so a hang can't run overnight, 15-s VRAM poller for the peak.
  If peak fits in GPU 2's ~97 GB free → train there. Result: **SUCCESS** —
  50 real-data steps, loss 4.22→3.66, ~1.0 s/step, exit 0. Peak VRAM 187 GB total
  − 101 GB co-tenant = **~86 GB true model footprint**. All previous fraction choices
  (.99/.90/.80/.70 = 204–144 GB) were massively oversized; the model fits GPU 2 today.
- **12:1x** FULL RUN LAUNCHED on GPU 2 (PID 2798803): proven diag-E config —
  `ROCBLAS_USE_HIPBLASLT=0`, command buffers off, fraction .44 (≈91 GB pool,
  preallocated to defend against co-tenant growth), compile cache persisted to
  `~/.jax_cache_gpu2` (13 MB, contains the verified autotuned executables;
  the old `~/.jax_cache` is poisoned — do not reuse). launch_train.sh defaults
  updated to this configuration. wandb tracking on.
- **12:03** Full run CRASHED — same rocBLAS internal error at first step, on the SAME
  GPU 2 where diag E had just succeeded. Two deviations from diag E: (1) preallocated
  .44 pool (91 GB) instead of on-demand → only ~6 GB raw VRAM left beside the 109 GB
  co-tenant; (2) compile cache missed (persisted 13 MB cache didn't include the big
  module), so autotune re-ran under that starved memory state. wandb `3bv9d0z6`.
  LESSON: promote a proven config verbatim; memory-allocator settings are part of it.
- **12:1x** Relaunched (PID 2807945) with the EXACT diag-E memory regime:
  `XLA_PYTHON_CLIENT_PREALLOCATE=false`, fraction .95, hipBLASLt off, GPU 2.
  Accepted trade-off: on-demand allocation means a co-tenant spike could OOM us
  mid-run; the monitor will catch it. Result: CRASHED in 97 s — but with NO compile
  phase: it cache-hit the executable that the 12:02 starved-memory run had compiled and
  written into `.jax_cache_gpu2`. The bad autotune choice was baked in. wandb `av8yu0nr`.
  MECHANISM CONFIRMED: executables autotuned under VRAM starvation carry a broken rocBLAS
  solution; the persistent cache then replays the crash regardless of runtime env.
- **12:1x** Purged `.jax_cache_gpu2`, relaunched (PID 2812399) with diag-E config and a
  clean compile under healthy memory (~97 GB free at autotune time). If this steps, the
  cache then holds a GOOD executable and future restarts are safe+fast. If it crashes,
  next lever: `--xla_gpu_autotune_level=0` (skip solution picking entirely).
  Result: **DEADLOCKED** — compile finished 12:17 (2m57s), then no output for 60+ min,
  140% CPU. Thread dump: one thread in `kfd_wait_on_events` (GPU event never signaled),
  one spinning, py_xla_execute parked on futex. Same class as the GPU-1 overnight hang.
  So the deadlock is NOT GPU-1-specific — it's intermittent on GPU 2 too; diag E just
  hit a good window. Killed 13:15. GPU 2 sanity gemm afterwards: instant OK.
- **13:2x** New best-candidate config: `--xla_gpu_autotune_level=0` added (no GEMM
  autotune probing at compile, no explicit solution replay at run — every failure so
  far implicates that machinery under co-tenant load). Expected cost: modestly slower
  steps. Fresh cache again (flag changes the key anyway). Relaunched (PID 2886641).
  Result: fast deterministic FAIL at first step — `CANCELLED: Contracting dimension is
  too fragmented.` With autotune off, XLA's heuristic routed a dot to the Triton GEMM
  emitter, which rejects that layout. wandb `k6crcws0`. Encouraging: no rocBLAS error,
  no hang — just a routing problem.
- **13:2x** Added `--xla_gpu_enable_triton_gemm=false` (all dots → plain rocBLAS default
  algorithm; no autotune, no solution replay, no Triton). Fresh cache, relaunched
  (PID 2891071). Result: DEADLOCKED again (kfd_wait_on_events + spinner) — this time
  BEFORE the compile alarm, i.e. the wedge point moves around. Conclusion: not a kernel-
  selection problem at all; it's HSA-level contention with the co-tenant. GPU 2's
  environment changed since diag E (~11:40, worked): every attempt from 12:00 on
  crashed or deadlocked while the tenant runs 100% util.
- **13:5x** Added `HSA_ENABLE_SDMA=0` (classic multi-process-on-one-GPU hang fix: stop
  using the shared SDMA copy engines, do copies via blit kernels). Kept autotune off +
  triton off. Relaunched (PID 2960992). Result: DEADLOCKED again, identical signature.
  SDMA is not (the whole) story. GPU 2 is consistently hostile since ~12:00 — its
  tenant's workload changed; diag E's 11:40 success was a lucky window.
- **14:2x** Switched to GPU 3 (~181 GB free, tenants compute-busy but memory-light) and
  added `GPU_MAX_HW_QUEUES=2` (queue oversubscription is the other documented cause of
  KFD event hangs on shared MI300X). Full hardened env now: hipBLASLt off, command
  buffers off, autotune off, Triton GEMM off, SDMA off, 2 HW queues, on-demand alloc.
  Launched PID 3652332. Killed after 30 min "no step" — see next entry.
- **14:5x** Diag-E replay on GPU 2 (cache hit, byte-identical env): WORKS, 1.4–1.8 s/step.
  Node did NOT change at noon. This prompted re-examination of the "deadlocks".
- **15:0x** **MAJOR CORRECTION — the afternoon "deadlocks" were never deadlocks.**
  Raw wandb event files of the killed runs: q65vg5p6 5.5 MB, zlc5hckv 7.9 MB,
  jjw3a9ff 8.0 MB of logged step metrics (thousands of steps) vs ~19 KB for the real
  crashes. The 12:08 (GPU 2, diag-E env), 13:15 (GPU 2, +SDMA off) and 14:18 (GPU 3,
  fully hardened) runs were all TRAINING AT FULL SPEED when killed. Root cause of the
  misdiagnosis: `print()` to a redirected log is block-buffered — nothing appears until
  the process exits (short diagnostics flushed at exit, so they "worked"). The
  `kfd_wait_on_events` thread is a normal idle JAX runtime thread, not a deadlock proof.
  Real failures remain: GPU 1 (crash w/ hipBLASLt, hang w/o — diag D never exited after
  50 steps in 14 h), and the three fast rocBLAS/Triton crashes (poisoned cache /
  starved-memory autotune / autotune-off Triton mis-routing).
  LESSONS: (1) `python -u` for any long run watched via log file; (2) check wandb event
  growth before declaring a hang; (3) a moving "wedge point" should have been the clue
  that observation, not execution, was broken.
- **15:1x** launch_train.sh: added `-u`. Relaunched full training on GPU 3 (PID 282366),
  identical env to the run that was healthy at 14:18. Result: **STEPPING** — visible in
  real time now: loss 4.40 → 3.76 over first 30 steps, ~1.4 s/step. wandb `utv8d2ru`.
  ~5000 steps/epoch → ≈2 h/epoch → README's ~120-epoch convergence ≈ 10 days.
  2-hourly monitor armed (checks log AND wandb event growth; prunes checkpoints if
  quota tightens; relaunches on crash with a 2-strikes rule).

## Planned fallbacks (in order)

1. Triton GEMM bypass (running).
2. `ROCBLAS_DEVICE_MEMORY_SIZE` preallocated workspace + rocBLAS trace logging
   (`ROCBLAS_LAYER=3`) to capture the exact failing GEMM dims.
3. Try another GPU with enough free VRAM (GPU 0: ~142 GB free but 99% busy compute;
   GPU 3: ~152 GB free, 50–91% busy; GPU 2: known-good but currently only ~90 GB free).
4. Reduced XLA pool + `XLA_PYTHON_CLIENT_PREALLOCATE=false` to measure true model footprint,
   enabling a move to a partially free GPU.
- **17:15** Monitor check #1: alive (PID 282366, 2h10m), ~3,760 steps, loss 4.40→~1.9, ~2.0 s/step avg (tenant contention varies 1.4–2.9 s). wandb event file growing (verified over sample window). Quota 171/200 GB, no checkpoints yet (first at epoch 5, ~11 h in). WATCH: GPU 3 co-tenant grew ~25→~91 GB; slack now ~29 GB — on-demand alloc means a further tenant spike could OOM us.
- **19:15** Monitor check #2: REAL STALL confirmed — log AND wandb console capture both
  frozen since 16:50 (step ~3,760, 1.5 h of healthy training); wandb file growth was
  telemetry only (~43 KB/min vs ~250 KB/min while stepping). The intermittent GPU wedge
  is real, it just strikes after hours, not minutes. Killed PID 282366.
- **19:3x** Countermeasures implemented: (1) `Trainer.restore()` + rolling resume
  checkpoints every 1000 steps (~40 min) in train.py — restarts now lose ≤1000 steps
  instead of everything (model + optimizer + LR-schedule step all restored; loader
  restarts at a fresh shuffled epoch, harmless); keeps newest 2 resume dirs. (2)
  `train_watchdog.sh` (detached, 5-min poll): relaunches on process death or >20-min
  log stall (won't judge runs <25 min old, gives up after >6 restarts/6 h). Relaunched
  training (PID 1286094) + watchdog (PID 1287265) on GPU 3. Liveness lesson updated:
  wandb FILE growth includes telemetry — only step-counter/console growth counts.
- **21:13** Monitor check #3: healthy. PID 1286094 up 1h55m, ~4,210 steps this run, loss ~1.9–2.1, ~1.4 s/step, log actively growing. Resume ckpts at steps 3000+4000 (406 MB total, pruning works). Quota 172/200 GB. GPU3 179/206 GB. Watchdog idle. wandb run: gw6wb9w6.
- **23:14** Monitor check #4: healthy. PID 1286094 up 3h55m (longest run yet), ~8,770 steps, loss ~1.6–1.9 (slow steady decline), 1.4 s/step, log live. Epoch 1 underway. Resume ckpts 7000+8000. Quota 172/200 GB. Watchdog idle.

## 2026-07-09
- **01:14** Monitor check #5: healthy. PID 1286094 up 5h55m, ~13,290 steps (epoch 2), loss ~1.7–2.0, 1.4 s/step, log live. Resume ckpts 12000+13000. Quota 172/200 GB. Watchdog idle.
- **03:14** Monitor check #6: healthy. PID 1286094 up 7h55m, ~17,830 steps (epoch 3), loss ~1.7–2.0, 1.4 s/step, log live. Resume ckpts 16000+17000. Quota 172/200 GB. Watchdog idle.
- **05:14** Monitor check #7: watchdog VALIDATED overnight — 04:19 stall (9-h run wedged ~step 20,400), auto-kill + relaunch, resumed from ckpt step 19000; now ~step 20,900, loss ~1.7–2.0, 1.4 s/step, wandb run no23vfg4. Net wedge cost ~35 min. WATCH: GPU 3 now 206/206 GB (co-tenant grew ~27 GB) — our allocation is held safe, but a restart while the tenant occupies our share could OOM-thrash; watchdog's 6-per-6h give-up rule guards it.
- **07:14** Monitor check #8: healthy + MILESTONE — first epoch checkpoint saved (me-full_25000_5, epoch 5). PID 1536351 up 2h54m, ~step 25,400, loss ~1.6–1.9, 1.4 s/step, log live. Resume ckpts 24000+25000. Quota 172/200 GB. GPU3 still full (206/206) but stable.
- **09:19** Monitor check #9: healthy. PID 1536351 up 5h, ~step 30,100 (epoch 6), loss ~1.7–1.9, 1.4 s/step, log live. Resume ckpts 29000+30000. Quota 174/200 GB (test set + sim assets added ~2 GB). Eval setup in progress on GPU 2 (test.zip extracted, YCB/GSO assets in, cv2-headless + MUJOCO_GL fixes; bench smoke running).
- **10:0x** EVAL PIPELINE COMPLETE (GPU 2): test.zip extracted (10 scenes × 5 grippers), 146 needed YCB/GSO objects selectively extracted (1.6 GB), three headless-node fixes: opencv-python→headless (GUI cv2 segfaults), MUJOCO_GL unset (EGL segfaults, OSMesa absent, GLFW aborts), MGS_NO_RENDER=1 + mgs/env/base.py patch to skip mujoco.Renderer (bench is physics-only). launch_bench.sh = proven env, GPU 2 default. Smoke: epoch-5 ckpt, panda scene_0008 → SR 99%, NJD 0.31.
- **11:13** Monitor check #10: healthy. PID 1536351 up 6h53m, ~step 34,400 (epoch 6), loss ~1.7–1.8, 1.4 s/step, log live. Resume ckpts 33000+34000. Quota 174/200 GB. Watchdog idle since 04:19. Epoch-5 eval (GPU 2) completed by user: SR panda 94.9% / vx300 93.3% / dexee 65.7% / allegro 81.3% / shadow 75.3%, mean 82.1%.
- **13:14** Monitor check #11: healthy. PID 1536351 up 8h54m, ~step 39,000 (epoch 7–8), loss ~1.65–1.72 (new low band), 1.4 s/step, log live (caught step-39000 resume save mid-write, normal). Quota 174/200 GB. Watchdog idle 9 h.
- **15:14** Monitor check #12: healthy. PID 1536351 up 10h54m, ~step 43,500 (epoch 8–9), loss ~1.7–1.9, 1.4 s/step, log live. Resume ckpts 42000+43000. Quota 174/200 GB. Watchdog idle ~11 h. Epoch-10 ckpt due ~step 50,000.
- **17:14** Monitor check #13: healthy. PID 1536351 up 12h54m, ~step 47,500 (epoch 9), loss ~1.8–2.0 in this sample (step noise; trend flat-to-down), step time slightly up (1.55–1.75 s, tenant contention), log live. Quota 174/200 GB. Watchdog idle 13 h. Epoch-10 ckpt ~1 h away.
- **19:14** Monitor check #14: healthy + MILESTONE — epoch-10 checkpoint saved (me-full_25000_10). PID 1536351 up 14h54m, ~step 51,000, loss lows now ~1.60, 1.4–1.6 s/step, log live. Quota 174/200 GB. Watchdog idle 15 h.
- **12:00** Monitor check #15: RESTART RECOVERED. Old PID 1536351 stalled ~06:57 (log idle 24+ min), watchdog killed+relaunched. New PID 2114378 (5h old) resumed from step-86000 checkpoint, now ~step 88,500 (epoch 17–18), loss ~1.72–1.89, 1.4 s/step, healthy. Epoch-15 checkpoint saved at step 75,000. Quota 175/200 GB. Restart #2 is first in current 6h window; continuing.
- **12:03** Monitor check #16: healthy, continuous. PID 2114378 up 5h04m, ~step 89,000 (epoch 18), loss ~1.70–1.79, 1.4 s/step, log live. Quota 175/200 GB.
- **15:04** Monitor check #17: healthy. PID 2114378 up 8h06m, ~step 93,500 (epoch 18–19), loss ~1.65–1.92, 1.4 s/step, log live. Resume ckpts at step92k/93k. Quota 175/200 GB.
- **20:05** Monitor check #18: healthy + MILESTONE — epoch-20 checkpoint saved. PID 2114378 up 13h07m, ~step 104,500 (epoch 20–21), loss ~1.63–1.89, 1.4 s/step, log live. Resume ckpts at step103k/104k. Quota 175/200 GB. GPU VRAM usage 156 GB.
- **20:07** Monitor check #19: healthy, continuous. PID 2114378 up 13h09m, ~step 104,600 (epoch 21), loss ~1.66–1.95, 1.4 s/step, log live. Quota 175/200 GB.
- **12:03** Monitor check #20: healthy + MAJOR MILESTONE — epoch-25 checkpoint saved (me-full_25000_25). PID 2114378 up 29h05m, ~step 140,000 (epoch 28), loss new low band ~1.68–1.75, 1.4 s/step, log live. Resume ckpts step139k/140k. Quota 176/200 GB. Watchdog idle >24h. Halfway to epoch-60 target.
- **12:04** Monitor check #21: healthy. PID 2114378 up 29h06m, ~step 140,100 (epoch 28, ~23% toward epoch 120), loss ~1.73–1.99, 1.4 s/step, log live. Quota 176/200 GB.
- **14:38** Monitor check #22: healthy + MILESTONE — epoch-30 checkpoint saved (me-full_25000_30). PID 2114378 up 31h40m, ~step 146,000 (epoch 29.2), loss improving ~1.54–1.72, 1.3 s/step, log live. Resume ckpts step145k/146k. Quota 176/200 GB. Epoch-25 eval running on GPU 2 (gripper 0 in progress).
- **15:47** Monitor check #23: healthy. PID 2114378 up 32h49m, ~step 148,000 (epoch 29.6), loss ~1.64–1.91, 1.4 s/step, log live. Resume ckpts step147k/148k. Quota 176/200 GB. Epoch-30 eval gripper 0 in progress (GPU 2).
- **17:24** Monitor check #24: healthy + EVAL COMPLETE — epoch-30 all-gripper eval done. PID 2114378 up 34h26m, ~step 152,000 (epoch 30.4), loss ~1.53–1.71, 1.4 s/step, log live. Resume ckpts step151k/152k. Quota 176/200 GB. Epoch-30 results: panda 96.0% / vx300 93.3% / dexee 76.0% / allegro 84.4% / shadow 71.6%, mean 84.3%.
- **17:40** TRAINING STOPPED BY USER REQUEST. Watchdog (PID 1287265) killed first to prevent auto-restart, then training PID 2114378 sent SIGTERM, exited cleanly. Last epoch checkpoint: me-full_25000_30 (epoch 30). Last resume checkpoint: step 153000 (~epoch 30.6). Resume checkpoints preserved on disk for future continuation.
- **18:23** 3-GPU RUN LAUNCHED per user request. Config change: num_scenes_per_batch 5->15 (kin_flow/cli/config/train.yaml) to divide evenly across 3 devices (was blocking reshape crash at N_DEVICES=3, since 5 is prime and only divides cleanly by 1 or 5). Each of 3 GPUs (2,6,7 - confirmed idle, no co-tenants) gets identical proven 5-scene/1-per-gripper shard; effective global batch triples 5->15/step. LR schedule already flat at end_lr=8e-5 since step ~10000 (decay_steps=10000 in trainer.py), so batch change has no schedule interaction. Resumed cleanly from step153000 checkpoint (PID 3897313), wandb run ocledg0m. User set completion target = epoch 120 (authors' README 'reasonable convergence' point, not the 500-epoch config cap).
- **18:25** 3-GPU config VALIDATED and finalized. 3 consecutive clean steps (1.61-1.62s each) at all-100% GPU compute utilization (GPUs 2,6,7). Throughput ~2.6x single-GPU baseline (15 scenes/1.6s vs 5 scenes/1.4s). No headroom to push further without going compute-idle-inefficient (already saturated). Watchdog relaunched pointed at new log/GPU set (TRAIN_LOG=train_me-full_3gpu.log, TRAIN_GPU=2,6,7, PID 3908717). Target completion = epoch 120 (per user decision, matches authors' README convergence guidance, not the 500-epoch config cap). 3-hourly monitoring cadence begins now.
- **15:41 (2026-07-13)** TARGET REACHED — epoch 130 (step ~218,000), past the epoch-120 completion goal set by user. 3-GPU run (GPUs 2,6,7, num_scenes_per_batch=15) confirmed healthy throughout: log+wandb both growing, all 3 GPUs at 100% compute, loss in 1.5-1.8 band. Checkpoints saved every 5 epochs through 130 (95,100,105,110,115,120,125,130 all present). Stopping 3-hourly cron loop per stop condition; training itself left running (will continue toward 500-epoch config cap unless stopped) since no instruction to halt it was given.
- **17:17** BUG FOUND+FIXED: bench.py/get_model_from_checkpoint (trainer.py) crashed on all epoch-130 eval attempts — Orbax refused to restore a checkpoint saved with 3-device sharding metadata (from the new 3-GPU training) onto a single-GPU eval process ('Topology mismatch'). Fix: kin_flow/ctrl/trainer.py get_model_from_checkpoint now passes target=train_state.to_pure_dict() to checkpointer.restore() (StandardCheckpointer.restore signature takes 'target', not orbax.checkpoint.args) so Orbax remaps the saved 3-way sharding onto whatever devices are available. Verified with 1-scene panda smoke test on epoch-130 (98% SR). This fix is required for evaluating ANY checkpoint saved after the 3-GPU switch (epoch>=95); earlier checkpoints (<=epoch 30, single-GPU-saved) still restore fine either way. Full 5-gripper eval relaunched on GPU 3.
