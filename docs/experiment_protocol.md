# Experiment protocol

## Current scientific gate

Phase 1.5 precedes CUDA profiling and closed-loop evaluation. It uses real
PushT conditioning and compares inference methods at exactly matched denoiser
function-evaluation (NFE) budgets.

For each `k in {5, 6, 7, 8}`, compare:

- plain DDIM-k;
- fixed-interval transformation reuse with exactly k U-Net calls; and
- adaptive caching with exactly k U-Net calls.

Use full DDIM-10 as the common reference. Report actual calls executed for
every sample; a threshold's nominal or average skip rate is not sufficient to
claim a matched budget.

## Paired evaluation

Every cached run is paired with an uncached run using the same checkpoint,
observation history, scheduler, initial diffusion noise and environment seed.
The cache is the only changed variable.

Offline trace replay is explicitly labeled as a baseline-path approximation:
it can screen thresholds cheaply, but it cannot represent the changed scheduler
trajectory after a reused output. Any selected threshold is therefore rerun
online through the actual scheduler before action error is reported.

## Feasibility gate

Before closed-loop evaluation, capture full denoising traces on at least 100
real conditioning examples and require all of:

1. adaptive caching beats plain DDIM-k on action error for at least three of
   the four matched NFE budgets;
2. at least one task-tolerable threshold skips approximately 20% or more of
   the denoiser calls on real conditioning; and
3. the visual encoder does not exceed 60% of visual-policy inference time.

If any condition fails, record the negative result before trying block-level
caching or a DiT-based action policy.

## Baselines

- uncached policy at identical denoising-step count;
- reduced-step DDIM;
- compiled uncached model;
- fixed-interval residual reuse; and
- adaptive accumulated-error caching.

## Metrics

Report denoiser calls, skip fraction, median and p95 action-chunk latency,
first-action maximum error, full-chunk MSE, smoothness and constraint
violations. Closed-loop reports include task success with confidence intervals,
time to success, collisions and safety interventions.

Development measurements on CPU or Apple MPS are functional checks only.
Published timing results require synchronized measurements on the target CUDA
GPU, including warmup and explicit hardware/software identification.

Simulation success is not a latency benefit unless wall-clock delay is
explicitly coupled to physics or action execution. The primary comparison is
therefore quality or success at fixed NFE, with latency reported separately.

Closed-loop arms use common random numbers and paired seeds. Pre-register a
non-inferiority margin, analyze paired success disagreements, and report the
achieved minimum detectable effect rather than describing a small run as
"unchanged success."
