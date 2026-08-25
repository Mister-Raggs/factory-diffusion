# Factory Diffusion working agreement

Before changing this repository, read:

1. `docs/PROJECT_CONTEXT.md`
2. `docs/HARDENING_REVIEW.md`
3. `docs/experiment_protocol.md`
4. `docs/experiment2_protocol.md`
5. `docs/experiment2_diagnostic_protocol.md`
6. `docs/experiment3_protocol.md`

Experiment 1 is complete: adaptive residual caching failed its matched-NFE
gate and is not to be revived. Experiment 2's calibration-only search for
few-step DDIM schedules passed its frozen held-out gate at all four NFE
budgets. Experiment 3 then failed its closed-loop non-inferiority rule over 400
episodes: optimized-minus-standard pooled success was -1.5 points, with a
-14-point regression at NFE 5. Do not start CUDA, Robomimic, or Factory SRE
work. The currently authorized work is only Experiment 2D, the post-hoc
one-chunk rollout-sensitivity diagnostic frozen in
`docs/experiment2_diagnostic_protocol.md`. It cannot alter the decisions of
Experiments 2 or 3 or tune schedules on Experiment 3 outcomes.

Experiment 2D is now complete and stopped its one-chunk proxy: the optimized
schedules remained physically closer to DDIM-10 at every budget, including the
failed NFE-5 schedule. The scientific study is archived with no active
experiment. Future work requires a new explicit protocol and should live in a
separate autonomy-system project unless it directly extends this result.

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
