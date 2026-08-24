# Factory Diffusion working agreement

Before changing this repository, read:

1. `docs/PROJECT_CONTEXT.md`
2. `docs/HARDENING_REVIEW.md`
3. `docs/experiment_protocol.md`
4. `docs/experiment2_protocol.md`
5. `docs/experiment3_protocol.md`

Experiment 1 is complete: adaptive residual caching failed its matched-NFE
gate and is not to be revived. Experiment 2's calibration-only search for
few-step DDIM schedules passed its frozen held-out gate at all four NFE
budgets. The next authorized scientific step is a small paired closed-loop
PushT evaluation of standard versus optimized schedules. Do not start CUDA,
Robomimic, or Factory SRE work until closed-loop task success is established.
Experiment 3's schedules, seeds, metrics, and decision rule are frozen in
`docs/experiment3_protocol.md`; smoke seeds must never enter the evaluation.

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
