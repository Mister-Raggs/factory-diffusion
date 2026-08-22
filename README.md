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

## Experiment gates

1. Profile the uncached policy and confirm repeated U-Net calls dominate
   action-chunk latency.
2. Capture an uncached denoising trace using fixed observations, seeds and
   initial noise.
3. Measure input change, output change, learned sensitivity and
   transformation-vector drift step by step.
4. Compare uncached, reduced-step DDIM, `torch.compile`, fixed skipping and
   adaptive caching.
5. Only then train or evaluate in Factory SRE and on SO-101 hardware.

Cache state is reset for every action chunk. State must never cross policy
queries or episode boundaries.

## Attribution

The adaptive rule is inspired by *Less is Enough: Training-Free Video
Diffusion Acceleration via Runtime-Adaptive Caching* (Zhou et al., 2025),
arXiv:2507.02860. This repository implements an independent robot-policy
adapter and evaluation harness; it does not vendor the reference code.
