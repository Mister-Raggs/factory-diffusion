# Factory Diffusion project context

This file is the durable handoff for future work on Factory Diffusion. It
captures the decisions previously held in the Physical Sprint conversation and
keeps them beside the code. Read it before planning or implementing a phase.

## Current objective

Factory Diffusion is a matched-compute study of inference shortcuts for
diffusion-based robot policies. Experiment 1 is complete and killed adaptive
residual caching. Experiment 2 found that calibration-selected DDIM timesteps
preserve DDIM-10 actions better than standard few-step schedules offline.

The current question is:

> At the same 2--5 denoiser calls, can a timestep schedule selected on a small
> calibration split preserve DDIM-10 actions better than the standard DDIM-k
> schedule without retraining?

This follows directly from Experiment 1 rather than rescuing its failed cache
mechanism. Reduced-step DDIM remains the baseline. The current question is now
whether the offline action-fidelity improvement produces equal or better
closed-loop PushT task success at the same exact NFE.

## Repository topology and boundaries

### This repository

- GitHub: `Mister-Raggs/factory-diffusion`
- Purpose: cache implementation, policy adapters, reproducible benchmarks,
  reports, and later demo integration protocol.
- Default branch: `main`.

### Hackathon repository

- Upstream: `joshuajerin/nebius-hackathon`
- Purpose here: a possible downstream Factory SRE demonstration using its
  factory scene, service/docking task, scripted controller, success predicates,
  and visualization.
- It is not merged or copied into Factory Diffusion and is not a primary
  experimental environment.
- Any eventual integration should be a clean branch of the then-current
  hackathon `main` that installs Factory Diffusion as a dependency.

### Old local simulation work

- Local path at the time of handoff: `/Users/raghavkachroo/my-sim`
- Historical branch: `raghav/sim`
- This tree contains extensive dirty and untracked hackathon-era work.
- Do not clean it, commit it, merge it, or use it as the Factory Diffusion
  workspace. It may be inspected read-only for concepts such as safety and
  visualization.

### FastVideo

FastVideo motivated the user's prior EasyCache-style work, but Factory
Diffusion remains standalone. Do not integrate into or modify FastVideo unless
that becomes a separate explicit task.

## Completed work

Phase 1 was completed in commit `e61a6b3` using:

- LeRobot `0.4.4`;
- the official `lerobot/diffusion_pusht_keypoints` checkpoint;
- immutable checkpoint revision
  `58570fc39828d28efa5457aa297a52be27ac3a10`;
- a fully loaded 248.8M-parameter temporal U-Net;
- six paired CPU probes with equal conditioning, initial noise, and scheduler
  seeds; and
- exact zero-threshold equivalence.

The Phase 1 probe found 20--40% skip opportunities near threshold `0.15`, but
that result is not a performance or feasibility claim because:

- conditioning was sampled uniformly from `[-1, 1]`, not from the PushT
  observation manifold;
- only six samples were tested;
- error was not tied to a task-derived tolerance;
- the keypoint policy omits visual encoder cost; and
- reduced-step DDIM at the same denoiser-call budget was not compared.

The earlier decision `proceed-to-cuda-and-closed-loop-validation` is superseded
by the Phase 1.5 gate below.

## Hardening decision

The full review is preserved in `docs/HARDENING_REVIEW.md`. Its binding changes
are:

1. Compare methods at matched denoiser function evaluations (NFE).
2. Use real PushT conditioning before interpreting residual stability.
3. Measure the visual encoder/U-Net latency split before claiming useful
   end-to-end acceleration.
4. Gate all credit expenditure on the zero-credit Phase 1.5 result.
5. Keep Factory SRE as motivation and a final visual demo, not as evidence for
   the scientific claim.

## Phase 1.5 protocol and completed result

Phase 1.5 completed with 100 real conditioning samples: 25 for calibration and
75 held out for evaluation. Across exact NFE budgets 5, 6, 7, and 8, plain
DDIM-k had lower mean first-action error and lower action-chunk MSE than fixed
or adaptive residual reuse at every budget. Adaptive caching won zero of four
budgets.

This fails the precommitted requirement to win at least three budgets. The
adaptive residual-caching acceleration thesis is stopped. Preserve the result
as a clean negative; do not proceed to CUDA, closed-loop, Robomimic, or Factory
SRE work for this mechanism. See `reports/phase15/SUMMARY.md`.

### A. Real conditioning

Load at least 100 valid conditioning examples from the public PushT dataset or
local PushT rollouts. Preserve real observation stacking and normalization.
Where possible, stratify samples into early, approach, and contact phases.
Repeat the threshold and denoising-trace analysis.

### B. Matched-NFE comparison

Use full DDIM-10 as the reference trajectory. For each budget
`k in {5, 6, 7, 8}`, compare methods that each execute exactly `k` U-Net calls:

1. plain DDIM-k;
2. fixed-interval transformation reuse; and
3. adaptive caching tuned or scheduled to land on k.

Measure full action-chunk MSE and first-executed-action error against DDIM-10.
Record exact NFE rather than inferring it from a requested threshold.

### C. Visual-policy Amdahl check

At the visual policy's real input shape, measure a ResNet-18 encoder forward
separately from ten temporal U-Net calls. Local CPU numbers are diagnostic; an
eventual CUDA claim requires warm-up, synchronization, and hardware/software
identification.

### D. Downstream feasibility

Test released Robomimic checkpoint compatibility before budgeting retraining.
This is a feasibility check, not authorization to begin Robomimic evaluation.

## Precommitted kill criteria

Stop the adaptive-caching acceleration thesis if any of the following holds:

- adaptive caching does not beat plain DDIM-k on action error for at least
  three of the four NFE budgets;
- real-conditioning skip fractions remain below approximately 20% at every
  threshold satisfying a task-derived first-action tolerance; or
- the visual encoder exceeds 60% of end-to-end visual-policy inference time,
  making denoiser caching the wrong optimization target.

If the thesis stops, preserve the harness and report the matched-budget negative
result. Do not rescue it by adding unrelated mechanisms or a bespoke task.

## Work authorized after Experiment 2

Phase 1.5 did not authorize downstream work for residual caching. Experiment 2
is a separate schedule-optimization mechanism and passed its precommitted
offline gate at all four budgets. It authorizes only the next staged test:

1. A small paired closed-loop PushT calibration arm using common random
   numbers, followed by a frozen evaluation if the integration is sound.
2. Standard DDIM-k versus the already selected optimized schedules only; do
   not retune schedules on closed-loop evaluation episodes.
3. Exact NFE and task success as primary controls. Treat CPU timing as
   diagnostic until a later CUDA timing protocol is frozen.

CUDA profiling, Robomimic, and Factory SRE remain deferred until the
closed-loop result establishes that the offline gains matter to task success.

## Experiment 2 offline result

On the frozen 25-sample calibration and 75-sample held-out split, optimized
schedules beat standard DDIM-k on both mean action-chunk MSE and mean
first-action pixel error at every exact NFE budget from two through five. The
selected schedules are `(70, 0)`, `(80, 10, 0)`, `(90, 50, 10, 0)`, and
`(90, 70, 30, 10, 0)`. All paired 95% bootstrap confidence intervals for both
improvements exclude zero. See `reports/experiment2/SUMMARY.md`.

This supports schedule optimization as an offline approximation method. It
does not yet show better task success or wall-clock acceleration.

## Final deliverable

The current scientific deliverable combines a clean negative and a positive:
residual reuse loses to reduced-step DDIM at matched NFE, while selecting the
few-step DDIM timesteps on a small task-specific calibration split improves
offline action fidelity at the same NFE.

The visual deliverable, if the schedule method survives closed loop, is a
short factory service/docking demonstration with an overlay of NFE, cache
decisions, and task status. It illustrates the result but does not establish
it.

## Data and compute decisions

- Do not download large NVIDIA human-video datasets for the current question.
  Raw human video does not supply the robot actions required by this benchmark
  and would introduce a separate embodiment problem.
- Do not spend Antioch or Nebius credits before Phase 1.5 is evaluated.
- Keep checkpoints, datasets, generated traces, and rollout videos gitignored.
- Calibrate one small rollout arm before authorizing a full sweep.

## Attribution and framing

The implementation is inspired by EasyCache-style runtime-adaptive caching; it
does not claim invention of the public EasyCache work. Relevant direct and
adjacent prior art includes BAC, reduced-step DDIM, one-step/consistency
distillation, ActionCache, VLA-Cache, and Real-Time Chunking. The README and any
write-up must present the matched-NFE comparison as the reason for this study.
