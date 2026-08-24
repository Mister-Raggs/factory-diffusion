# Experiment 2 protocol: task-calibrated DDIM schedules

This protocol is frozen before the full Experiment 2 evaluation.

## Question

At exact denoiser budgets `k in {2, 3, 4, 5}`, can a timestep schedule selected
on real PushT calibration observations preserve DDIM-10 actions better than
LeRobot/Diffusers' standard DDIM-k schedule without retraining the policy?

## Fixed data split and pairing

- Use the same pinned 100 real PushT conditioning samples and seed zero as
  Experiment 1.
- Samples 0--24 are calibration-only.
- Samples 25--99 are held out until one schedule per budget is frozen.
- Every comparison uses identical conditioning, initial diffusion noise,
  checkpoint, and DDIM transition implementation.
- Full DDIM-10 is the common reference.

## Candidate schedules

For each budget, enumerate every strictly descending schedule of exactly `k`
timesteps drawn from `{0, 10, ..., 90}` and ending at zero. Add the configured
standard Diffusers DDIM-k schedule if it is not already in that set.

The explicit DDIM transition must reproduce the native standard scheduler in a
unit test before custom schedules are evaluated.

## Calibration selection

Select one schedule per budget using calibration mean normalized action-chunk
MSE against DDIM-10. Break ties using calibration mean first-action pixel error
and then lexicographic schedule order. Held-out errors must never influence
selection or candidate revision.

## Held-out metrics

Report, for standard and optimized schedules:

- exact NFE and selected timesteps;
- mean and maximum first-executed-action error in PushT pixels;
- mean normalized full action-chunk MSE; and
- paired 95% bootstrap intervals for optimized-minus-standard differences.

CPU timings are diagnostic and are not performance claims.

## Precommitted gate

An optimized schedule wins a budget only if it has both lower held-out mean
action-chunk MSE and no higher held-out mean first-action pixel error than the
standard schedule. Proceed to paired closed-loop PushT evaluation only if the
optimized schedules win at least three of four budgets.

If the gate fails, preserve the result and stop this schedule-optimization
branch. Do not change the grid, split, objective, or gate after seeing held-out
results.

## Recorded outcome

The full run completed under this protocol and passed the gate at all four
budgets. The selected schedules were `(70, 0)`, `(80, 10, 0)`,
`(90, 50, 10, 0)`, and `(90, 70, 30, 10, 0)` for NFE 2 through 5. The next
authorized step is paired closed-loop PushT evaluation; the candidate grid,
split, and selected schedules remain frozen. See
`reports/experiment2/SUMMARY.md` for the held-out measurements.
