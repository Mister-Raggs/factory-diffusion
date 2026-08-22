# Experiment protocol

## Paired evaluation

Every cached run is paired with an uncached run using the same checkpoint,
observation history, scheduler, initial diffusion noise and environment seed.
The cache is the only changed variable.

Offline trace replay is explicitly labeled as a baseline-path approximation:
it can screen thresholds cheaply, but it cannot represent the changed scheduler
trajectory after a reused output. Any selected threshold is therefore rerun
online through the actual scheduler before action error is reported.

## Feasibility gate

Before closed-loop evaluation, capture full denoising traces and require both:

1. enough predicted-error intervals to skip repeated U-Net evaluations; and
2. enough U-Net share of end-to-end latency for those skips to matter.

If either condition fails, record the negative result before trying block-level
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
