# Factory Diffusion

Factory Diffusion studies whether runtime-adaptive residual caching can reduce
the inference latency of diffusion-based robot policies without materially
reducing closed-loop task success.

The first target is LeRobot's Diffusion Policy. Factory SRE, the hackathon
simulation, is an optional downstream environment for battery-service and
docking evaluation; it is deliberately not copied into this repository.

## Research question

Can an EasyCache-style accumulated-error rule safely reuse a denoiser's
transformation vector across adjacent action-denoising steps, and how does the
threshold move the latency-versus-success Pareto frontier?

This is a transfer study, not an assumption that video-model results apply to
robot policies. LeRobot's baseline uses a temporal 1D U-Net rather than a
Wan-style video DiT, so residual stability is tested before training or large
rollout sweeps.

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

The provisional result is to proceed to CUDA and closed-loop validation. With
two warmup steps, two forced final recomputations and at most two consecutive
skips, threshold `0.15` skipped 20–40% of the ten denoising calls across the
six probes. Its worst normalized first-action deviation was `0.03386`.

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

## Experiment gates

1. Profile the uncached policy and confirm repeated U-Net calls dominate
   action-chunk latency.
2. Capture an uncached denoising trace using fixed observations, seeds and
   initial noise.
3. Measure input change, output change, learned sensitivity and
   transformation-vector drift step by step.
4. Screen adaptive thresholds offline, then rerun selected thresholds online
   with the real scheduler.
5. In Phase 2, compare reduced-step DDIM, `torch.compile`, fixed skipping and
   adaptive caching on CUDA and in closed loop.
6. Only then train or evaluate in Factory SRE and on SO-101 hardware.

Cache state is reset for every action chunk. State must never cross policy
queries or episode boundaries.

## Attribution

The adaptive rule is inspired by *Less is Enough: Training-Free Video
Diffusion Acceleration via Runtime-Adaptive Caching* (Zhou et al., 2025),
arXiv:2507.02860. This repository implements an independent robot-policy
adapter and evaluation harness; it does not vendor the reference code.
