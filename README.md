# Factory Diffusion

Factory Diffusion studies **task-calibrated few-step sampling for diffusion
robot policies**. Given a fixed budget of denoiser calls, it selects the DDIM
timesteps that best preserve a pretrained policy's action trajectory—without
retraining or changing the model.

The first target is LeRobot's keypoint-conditioned PushT Diffusion Policy. The
current result is offline; paired closed-loop PushT evaluation is next.

## Current result

Experiment 2 selected one schedule per exact denoiser function-evaluation
(NFE) budget using 25 real PushT calibration observations, then evaluated the
frozen schedules on 75 held-out observations. Optimized schedules beat standard
DDIM-k on both action-chunk MSE and first-executed-action error at all four
budgets:

| NFE | Standard schedule | Optimized schedule | Standard first-action error (px) | Optimized (px) |
| ---: | :--- | :--- | ---: | ---: |
| 2 | `50, 0` | `70, 0` | 3.107 | **1.447** |
| 3 | `66, 33, 0` | `80, 10, 0` | 0.837 | **0.360** |
| 4 | `75, 50, 25, 0` | `90, 50, 10, 0` | 0.493 | **0.109** |
| 5 | `80, 60, 40, 20, 0` | `90, 70, 30, 10, 0` | 0.316 | **0.049** |

All paired 95% bootstrap intervals for both improvements exclude zero. This
passes the precommitted offline gate 4/4 and authorizes a paired closed-loop
PushT experiment.

This result does **not** yet establish better task success or wall-clock
acceleration. Both arms use the same exact NFE; CPU timings are diagnostic.

- [Experiment 2 summary](reports/experiment2/SUMMARY.md)
- [Frozen Experiment 2 protocol](docs/experiment2_protocol.md)
- [Machine-readable metrics](reports/experiment2/summary.json)

## Research question

> At the same small denoiser-call budget, can a timestep schedule selected on a
> small task-specific calibration split preserve full DDIM-10 behavior—and
> ultimately closed-loop task success—better than the standard DDIM-k schedule?

The method searches a finite timestep grid using a frozen pretrained policy.
It requires no gradient updates, policy demonstrations, or model distillation.
Reduced-step DDIM is always the matched-compute baseline.

## Why the project pivoted

Experiment 1 tested adaptive residual reuse inspired by diffusion-model caching.
The hardening pass introduced the missing matched-NFE baseline: simply running
a coherent shorter DDIM trajectory. On 75 held-out real PushT observations,
plain DDIM-k beat fixed and adaptive reuse at every tested NFE budget. The cache
mechanism therefore failed its frozen survival gate and was stopped.

That clean negative pointed to the useful variable: not whether denoiser work
can be reused, but **which timesteps deserve a limited number of calls**.
Experiment 2 follows that evidence rather than attempting to rescue caching.

- [Experiment 1 matched-NFE summary](reports/phase15/SUMMARY.md)
- [Historical hardening review](docs/HARDENING_REVIEW.md)

The cache implementation and earlier reports remain in the repository for
reproducibility, but they are no longer the active method or project claim.

## Roadmap

1. **Complete:** validate explicit DDIM transitions against the standard
   scheduler.
2. **Complete:** select schedules on calibration-only observations and run one
   frozen held-out offline evaluation.
3. **Next:** compare standard and optimized schedules in paired closed-loop
   PushT episodes using common random numbers and exact NFE accounting.
4. **Conditional:** profile CUDA wall-clock latency only if task success is
   retained.
5. **Optional demo:** integrate the surviving method into Factory SRE for a
   visual factory-service scenario. Factory SRE is not the scientific
   benchmark.

The optimized schedules, held-out split, and offline selection objective are
frozen. They must not be retuned on closed-loop evaluation episodes.

The paired design, 50 evaluation seeds, success criterion, and non-inferiority
gate are frozen in [the Experiment 3 protocol](docs/experiment3_protocol.md).
The headless checkpoint-to-controller-to-physics smoke test passes for both
arms, including full 300-step termination and exact NFE accounting.

## Repository layout

- `src/factory_diffusion/schedules.py`: explicit deterministic DDIM schedules
  and transitions.
- `src/factory_diffusion/schedule_search.py`: calibration selection and paired
  statistics.
- `src/factory_diffusion/evaluation.py`: exact-NFE samplers and evaluation
  helpers.
- `experiments/04_optimize_ddim_schedule.py`: Experiment 2 end-to-end runner.
- `experiments/05_closed_loop_pusht.py`: paired closed-loop Experiment 3 runner.
- `reports/experiment2`: durable Experiment 2 results.
- `src/factory_diffusion/cache` and `experiments/01_*` through `03_*`:
  preserved Experiment 1 implementation and evaluation.
- `src/factory_diffusion/integrations/factory_sre.py`: boundary for a possible
  later downstream demo.

Checkpoints, datasets, generated traces, raw reports, and videos stay out of
git. Nothing here vendors or modifies the hackathon repositories or FastVideo.

## Local setup

Python 3.10 matches the tested LeRobot 0.4.4 environment:

```bash
uv venv --python 3.10
source .venv/bin/activate
uv sync --extra dev --extra lerobot
python -m unittest discover -s tests
```

The official keypoint policy checkpoint is pinned at revision
`58570fc39828d28efa5457aa297a52be27ac3a10`; the PushT keypoint dataset is
pinned at revision `ace8c161a68bc025c21a5f29f85b86a9a2c5e64b`.

## Reproduce Experiment 2

The full offline run uses 100 deterministic real PushT observations, with 25
reserved for schedule selection and 75 held out:

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

`--local-files-only` prevents checkpoint network access. The pinned checkpoint
and dataset must already exist locally. The generated report records immutable
revisions, the sample manifest, every calibration candidate, exact NFE, paired
held-out rows, confidence intervals, and the frozen decision.

For a cheap integration check, reduce `--samples`, `--calibration-samples`, and
`--budgets`. Any noncanonical configuration is labeled `smoke-only` and cannot
produce a scientific proceed/stop decision.

## Run Experiment 3

The full paired closed-loop evaluation uses 50 environment seeds at all four
frozen budgets, for 400 total episodes:

```bash
SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy \
HF_HUB_OFFLINE=1 PYTHONPATH=src python experiments/05_closed_loop_pusht.py \
  --device cpu \
  --seeds 0:50 \
  --budgets 2,3,4,5 \
  --max-steps 300 \
  --local-files-only \
  --dataset-root data/pusht-keypoints \
  --cache-dir .cache/huggingface/hub \
  --output-dir outputs/experiment3/closed-loop
```

The runner atomically checkpoints after every episode. Restarting the same
command skips completed method/seed/budget cells, while a conflicting run
configuration is rejected. Generated partial and final reports remain
gitignored.

## Historical Experiment 1 reproduction

The earlier cache probes remain runnable for auditability:

```bash
PYTHONPATH=src python experiments/01_pusht_keypoints_probe.py \
  --device cpu \
  --force-compute-last 2 \
  --output-dir outputs/phase1/pusht-keypoints-safe-seed0

PYTHONPATH=src python experiments/03_phase15_matched_nfe.py \
  --samples 100 \
  --calibration-samples 25 \
  --budgets 5,6,7,8 \
  --local-files-only \
  --dataset-root data/pusht-keypoints \
  --cache-dir .cache/huggingface/hub \
  --output-dir outputs/phase15/matched-nfe
```

See [the durable project context](docs/PROJECT_CONTEXT.md) before extending the
study. It records repository boundaries, stopped hypotheses, and the current
authorization gate.

## Attribution

Experiment 1's adaptive rule was inspired by *Less is Enough: Training-Free
Video Diffusion Acceleration via Runtime-Adaptive Caching* (Zhou et al., 2025),
arXiv:2507.02860. Factory Diffusion implements an independent robot-policy
adapter and evaluation harness and does not vendor the reference code.
