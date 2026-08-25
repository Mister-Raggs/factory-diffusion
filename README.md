# Factory Diffusion

Factory Diffusion studies **task-calibrated few-step sampling for diffusion
robot policies**. Given a fixed budget of denoiser calls, it selects the DDIM
timesteps that best preserve a pretrained policy's action trajectory—without
retraining or changing the model.

The first target is LeRobot's keypoint-conditioned PushT Diffusion Policy. The
completed study shows why offline denoising fidelity—even after one chunk of
physics—is not sufficient evidence for closed-loop robot performance.

## Current result

The evidence forms a three-level proxy ladder:

1. **Experiment 2 — offline fidelity:** optimized schedules approximate
   DDIM-10 actions better at every NFE budget from two through five.
2. **Experiment 2D — one-chunk physics:** across 244 DDIM-10 teacher states
   and 1,952 counterfactual branches, optimized schedules also produce lower
   post-chunk physical divergence at every budget, including 169 contact
   states.
3. **Experiment 3 — complete closed loop:** across 400 episodes, pooled
   optimized success is 70.5% versus 72.0% standard. At NFE 5 it falls from
   78% to 64%, failing the frozen non-inferiority rule.

| NFE | Standard success | Optimized success | Optimized minus standard |
| ---: | ---: | ---: | ---: |
| 2 | 70% | 70% | 0 points |
| 3 | 68% | 74% | +6 points |
| 4 | 72% | 74% | +2 points |
| 5 | 78% | 64% | **-14 points** |

The result is a clean negative for the selection objective: local agreement
with a full-compute teacher does not reliably predict task success after the
policy repeatedly acts on states induced by its own decisions. Both methods
use the same exact NFE, so this is not a wall-clock acceleration claim.

- [Experiment 2 summary](reports/experiment2/SUMMARY.md)
- [Experiment 2D rollout-sensitivity diagnostic](reports/experiment2/ROLLOUT_SENSITIVITY.md)
- [Experiment 3 closed-loop summary](reports/experiment3/SUMMARY.md)
- [Complete results index](reports/RESULTS.md)
- [Frozen Experiment 2 protocol](docs/experiment2_protocol.md)
- [Experiment 2D diagnostic protocol](docs/experiment2_diagnostic_protocol.md)
- Machine-readable metrics: [Experiment 2](reports/experiment2/summary.json),
  [Experiment 2D](reports/experiment2/rollout_sensitivity_summary.json), and
  [Experiment 3](reports/experiment3/summary.json)

## Research question

> At the same small denoiser-call budget, which offline or short-horizon proxy,
> if any, reliably selects a DDIM timestep schedule for closed-loop control?

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
3. **Complete:** evaluate the frozen schedules over 400 paired closed-loop
   PushT episodes. The method failed its non-inferiority rule.
4. **Complete diagnostic:** test one-chunk physical sensitivity along DDIM-10
   trajectories. This proxy also favored the failed NFE-5 schedule and was
   stopped before schedule search.
5. **Stopped:** CUDA claims and Factory SRE integration are not authorized by
   the completed results. A future extension requires multi-query or on-policy
   calibration and a fresh held-out evaluation protocol.

## Repository layout

- `src/factory_diffusion/schedules.py`: explicit deterministic DDIM schedules
  and transitions.
- `src/factory_diffusion/schedule_search.py`: calibration selection and paired
  statistics.
- `src/factory_diffusion/evaluation.py`: exact-NFE samplers and evaluation
  helpers.
- `experiments/04_optimize_ddim_schedule.py`: Experiment 2 end-to-end runner.
- `experiments/05_closed_loop_pusht.py`: paired closed-loop Experiment 3 runner.
- `src/factory_diffusion/rollout_sensitivity.py`: exact-state one-chunk
  counterfactual branches and seed-balanced diagnostic summaries.
- `experiments/06_rollout_sensitivity_diagnostic.py`: resumable Experiment 2D
  runner.
- `reports/experiment2`: durable Experiment 2 results.
- `reports/experiment3`: durable closed-loop result.
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
gitignored. The frozen run completed with decision `inconclusive-or-negative`;
see [the durable summary](reports/experiment3/SUMMARY.md).

## Run the Experiment 2D diagnostic

The post-hoc diagnostic restores exact DDIM-10 teacher body states and executes
one paired eight-action physics branch for each frozen schedule:

```bash
SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy \
HF_HUB_OFFLINE=1 PYTHONPATH=src python \
  experiments/06_rollout_sensitivity_diagnostic.py \
  --device cpu \
  --seeds 0:10 \
  --budgets 2,3,4,5 \
  --max-steps 300 \
  --local-files-only \
  --dataset-root data/pusht-keypoints \
  --cache-dir .cache/huggingface/hub \
  --output-dir outputs/experiment2/rollout-sensitivity
```

The run checkpoints after each teacher seed. It is exploratory and cannot
change Experiment 2 or 3; its completed `stop-proxy` result is summarized in
[the diagnostic report](reports/experiment2/ROLLOUT_SENSITIVITY.md).

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
