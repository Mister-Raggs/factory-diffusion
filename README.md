# Factory Diffusion

Factory Diffusion is a matched-compute study of inference shortcuts for
diffusion-based robot policies. It asks whether adaptive caching preserves
action quality and closed-loop task success better than simply running fewer
DDIM steps at the same denoiser function-evaluation budget.

The first target is LeRobot's Diffusion Policy. Factory SRE, the hackathon
simulation, is an optional downstream environment for battery-service and
docking evaluation; it is deliberately not copied into this repository.

## Research question

At a fixed budget of denoiser function evaluations, which method best preserves
the full DDIM-10 policy's actions and closed-loop success: reduced-step DDIM,
fixed-interval reuse, or adaptive residual caching?

Direct caching prior art for Diffusion Policy already exists. The purpose of
this repository is therefore not to claim that caching transfers to robotics,
but to run the matched-budget comparison that determines whether caching buys
anything beyond fewer denoising steps.

Read [the durable project context](docs/PROJECT_CONTEXT.md) and
[the hardening review](docs/HARDENING_REVIEW.md) before extending the project.

## Repository boundaries

- `factory_diffusion.cache`: framework-neutral cache state and decisions.
- `factory_diffusion.integrations.lerobot`: a small PyTorch denoiser wrapper;
  LeRobot itself stays an external pinned dependency.
- `factory_diffusion.integrations.factory_sre`: the interface that a later
  integration branch in the main Factory SRE repository will implement.
- `experiments`: profiling and feasibility probes. Generated data and
  checkpoints stay out of git.

Nothing here modifies or vendors the hackathon repositories or FastVideo.

## Phase 1 status

Phase 1 is complete on the pinned public keypoint-conditioned PushT policy:

- exact revision: `58570fc39828d28efa5457aa297a52be27ac3a10`;
- real 248.8M-parameter temporal U-Net, not a synthetic model;
- six paired seeds with identical conditioning, initial noise and scheduler
  seeds;
- uncached tensor traces and baseline-path threshold replay;
- exact online cached runs that include scheduler-path divergence; and
- zero-threshold equivalence verified for every seed.

With two warmup steps, two forced final recomputations and at most two
consecutive skips, threshold `0.15` skipped 20–40% of the ten denoising calls
across the six probes. Its worst normalized first-action deviation was
`0.03386`.

That result is now treated only as a mechanism smoke test. The conditioning was
synthetic and out of distribution, the sample was small, and no matched-NFE
DDIM baseline was run. The old provisional decision to proceed directly to
CUDA and closed-loop validation has been superseded by Phase 1.5.

See [the Phase 1 summary](reports/phase1/SUMMARY.md) and its machine-readable
[metrics](reports/phase1/summary.json). CPU timings in that report are strictly
functional measurements, not performance claims.

## Local setup

Python 3.10 is used initially to match the already-tested local LeRobot 0.4.4
environment. Create a clean environment when dependency installation is
available:

```bash
uv venv --python 3.10
source .venv/bin/activate
uv sync --extra dev --extra lerobot
python -m unittest discover -s tests
```

The cache core can also be tested using the existing hackathon LeRobot
environment without installing anything into that repository:

```bash
PYTHONPATH=src /path/to/python -m unittest discover -s tests
PYTHONPATH=src /path/to/python experiments/00_smoke_cache.py
```

## Reproduce the Phase 1 probe

The official visual `lerobot/diffusion_pusht` checkpoint has a known image
encoder compatibility problem with LeRobot 0.4.4. Phase 1 therefore uses the
official keypoint-conditioned checkpoint, which exercises the same action
U-Net without the incompatible image encoder. The loader translates its legacy
configuration and pins its immutable Hub revision.

Run a single seed:

```bash
PYTHONPATH=src python experiments/01_pusht_keypoints_probe.py \
  --device cpu \
  --force-compute-last 2 \
  --output-dir outputs/phase1/pusht-keypoints-safe-seed0
```

Aggregate a set of seed reports:

```bash
PYTHONPATH=src python experiments/02_summarize_phase1.py outputs/phase1 \
  --path-glob 'pusht-keypoints-safe-seed*/report_seed_*.json' \
  --output-dir reports/phase1
```

The probe downloads the roughly 1 GB checkpoint into the gitignored local
cache on first use. Generated traces and outputs are also gitignored.

## Phase 1.5 gate and result

No project credits should be spent until these zero-credit checks complete:

1. Repeat the trace and threshold study on at least 100 real PushT conditioning
   examples sampled across the temporal span of episodes.
2. At each budget `k in {5, 6, 7, 8}`, compare plain DDIM-k,
   fixed-interval reuse, and adaptive caching using exactly k U-Net calls.
3. Measure the visual encoder/U-Net latency split at the real input shape.
4. Check whether released Robomimic checkpoints load without retraining.

Adaptive caching proceeds only if it beats DDIM-k on action error for at least
three of four budgets, retains meaningful safe skip opportunities on real
conditioning, and targets a material share of visual-policy latency. Otherwise
the matched-budget negative result is the deliverable.

CUDA and closed-loop evaluation come after this gate. Factory SRE remains an
optional final demonstration rather than an experimental benchmark.

Cache state is reset for every action chunk. State must never cross policy
queries or episode boundaries.

### Implementation status

The Phase 1.5 offline harness is implemented and locally validated:

- the official `lerobot/pusht_keypoints` dataset is pinned at revision
  `ace8c161a68bc025c21a5f29f85b86a9a2c5e64b`;
- the official keypoint policy checkpoint is pinned at revision
  `58570fc39828d28efa5457aa297a52be27ac3a10`;
- its 995,056,568-byte safetensors file matches SHA-256
  `ae40aa87dd124ee1e4914258049f4eac676345a28e6cb8dfcfa67830cc3246b0`;
- the checkpoint loads offline as a 248,759,426-parameter DDIM-10 policy; and
- the unit, lint, formatting, and compilation checks pass in the standalone
  Factory Diffusion environment.

The end-to-end smoke run passes, and the full 100-sample matched-NFE evaluation
is complete. It uses 25 calibration samples and 75 held-out samples, producing
900 method/sample/budget comparisons with exact per-run NFE.

| NFE | DDIM-k first-action error (px) | Fixed reuse (px) | Adaptive reuse (px) |
| ---: | ---: | ---: | ---: |
| 5 | **0.316** | 2.330 | 3.833 |
| 6 | **0.260** | 0.885 | 1.505 |
| 7 | **0.157** | 0.507 | 0.609 |
| 8 | **0.137** | 0.251 | 0.275 |

Plain DDIM-k also has the lowest action-chunk MSE at every budget. Adaptive
caching therefore wins zero of four budgets and fails the precommitted
three-of-four survival criterion. The acceleration thesis stops here: CUDA,
closed-loop, and Factory SRE work are not justified for this mechanism. The
clean negative result is the deliverable.

See [the Phase 1.5 summary](reports/phase15/SUMMARY.md) and its machine-readable
[metrics](reports/phase15/summary.json). These are offline action-fidelity
results against DDIM-10, not closed-loop task-success measurements.

The real-conditioning and matched-NFE harness is implemented in
`experiments/03_phase15_matched_nfe.py`. It pins the official PushT keypoint
dataset, samples temporal thirds of episodes deterministically, calibrates an
adaptive threshold without looking at evaluation action error, and evaluates
DDIM-k, fixed reuse, and adaptive reuse on the same held-out observations and
initial noise.

After activating the project environment, run a small integration smoke test:

```bash
source .venv/bin/activate
PYTHONPATH=src python experiments/03_phase15_matched_nfe.py \
  --samples 3 \
  --calibration-samples 1 \
  --budgets 5 \
  --local-files-only \
  --dataset-root data/pusht-keypoints \
  --cache-dir .cache/huggingface/hub \
  --output-dir outputs/phase15/smoke
```

Once that passes, run the full 100-sample gate:

```bash
PYTHONPATH=src python experiments/03_phase15_matched_nfe.py \
  --samples 100 \
  --calibration-samples 25 \
  --budgets 5,6,7,8 \
  --local-files-only \
  --dataset-root data/pusht-keypoints \
  --cache-dir .cache/huggingface/hub \
  --output-dir outputs/phase15/matched-nfe
```

`--local-files-only` prevents checkpoint network access. The pinned dataset
must already exist at `--dataset-root` for an offline rerun. The report records
the immutable dataset and checkpoint revisions, exact NFE for every run,
normalized action error, and action error converted back to PushT pixels.
Temporal thirds are sampling strata, not verified approach/contact labels.

## Experiment 2: Task-calibrated DDIM schedules

Experiment 1 showed that a coherent shorter DDIM trajectory preserves actions
better than residual reuse. Experiment 2 asks the direct follow-up:

> At the same 2--5 denoiser calls, can a timestep schedule selected on a small
> calibration split preserve DDIM-10 actions better than LeRobot's standard
> DDIM-k schedule, without changing policy weights?

The implementation enumerates schedules on the training-timestep grid
`{0, 10, ..., 90}`, adds the standard Diffusers schedule as a candidate, and
selects one schedule per budget using only 25 calibration observations. The
four schedules are frozen before a paired evaluation on the existing 75
held-out observations.

The project proceeds to closed-loop PushT only if optimized schedules lower
held-out mean action-chunk MSE without increasing mean first-action pixel error
at three of four budgets. The full offline experiment passed this gate at all
four budgets:

| NFE | Standard schedule | Optimized schedule | Standard first-action error (px) | Optimized (px) |
| ---: | :--- | :--- | ---: | ---: |
| 2 | `50, 0` | `70, 0` | 3.107 | **1.447** |
| 3 | `66, 33, 0` | `80, 10, 0` | 0.837 | **0.360** |
| 4 | `75, 50, 25, 0` | `90, 50, 10, 0` | 0.493 | **0.109** |
| 5 | `80, 60, 40, 20, 0` | `90, 70, 30, 10, 0` | 0.316 | **0.049** |

The optimized schedules also lowered mean action-chunk MSE at every budget.
All paired 95% bootstrap intervals for both reported improvements exclude
zero. This is an offline action-fidelity result, not yet a task-success or
wall-clock acceleration claim. The next authorized experiment is a paired
closed-loop PushT evaluation of standard versus optimized schedules.

See [the frozen Experiment 2 protocol](docs/experiment2_protocol.md) and
[the Experiment 2 result](reports/experiment2/SUMMARY.md).

Reproduce the full offline experiment with:

```bash
PYTHONPATH=src python experiments/04_optimize_ddim_schedule.py \
  --device cpu \
  --samples 100 \
  --calibration-samples 25 \
  --budgets 2,3,4,5 \
  --grid-step 10 \
  --batch-size 3 \
  --local-files-only \
  --dataset-root data/pusht-keypoints \
  --cache-dir .cache/huggingface/hub \
  --output-dir outputs/experiment2/schedules
```

## Attribution

The adaptive rule is inspired by *Less is Enough: Training-Free Video
Diffusion Acceleration via Runtime-Adaptive Caching* (Zhou et al., 2025),
arXiv:2507.02860. This repository implements an independent robot-policy
adapter and evaluation harness; it does not vendor the reference code.
