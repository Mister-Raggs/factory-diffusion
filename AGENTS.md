# Factory Diffusion working agreement

Before changing this repository, read:

1. `docs/PROJECT_CONTEXT.md`
2. `docs/HARDENING_REVIEW.md`
3. `docs/experiment_protocol.md`

The current scientific gate is Phase 1.5: a real-conditioning,
matched-function-evaluation comparison between reduced-step DDIM,
fixed-interval reuse, and adaptive caching. Do not start CUDA, closed-loop,
Robomimic, or Factory SRE work until the Phase 1.5 kill criteria have been
evaluated.

Repository boundaries:

- Do not merge, vendor, or modify either hackathon repository from here.
- Treat `joshuajerin/nebius-hackathon` as a read-only downstream demo source.
- Treat the old local `raghav/sim` work as a read-only donor of concepts only.
- Factory SRE is a later demonstration, not an experimental benchmark.
- Generated datasets, checkpoints, traces, and videos remain outside git.
- Never push to another project's `main`; changes for this project belong in
  `Mister-Raggs/factory-diffusion`.

Preserve paired evaluation: identical checkpoint, conditioning, initial noise,
scheduler, and environment seed, with the inference method as the only changed
variable. Report negative results and achieved statistical power explicitly.
