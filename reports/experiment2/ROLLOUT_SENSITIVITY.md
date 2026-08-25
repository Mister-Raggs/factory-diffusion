# Experiment 2D: one-chunk rollout-sensitivity diagnostic

## Outcome

**Decision: stop this proxy before rollout-aware schedule search.**

The diagnostic does not explain Experiment 3's NFE-5 regression. Optimized
schedules remained closer to DDIM-10 than standard schedules after one full
eight-action physics branch at every budget—including NFE 5 and contact
states—even though the optimized NFE-5 schedule later reduced complete-episode
success by 14 percentage points.

## Design

- Policy and checkpoint: identical to Experiments 2 and 3.
- Teacher: DDIM-10.
- Teacher seeds: `0..9`; this is post-hoc use of Experiment 3 seeds.
- Teacher query states: 244 total, including 169 with contact during the next
  action chunk.
- Branches: 1,952 paired standard/optimized counterfactuals.
- Pairing: exact teacher simulator body state and velocities, two-frame policy
  history, and initial diffusion noise.
- Horizon: one complete eight-action chunk.

Generated row-level results remain gitignored at
`outputs/experiment2/rollout-sensitivity/report.json`.

## Results

Differences below are optimized minus standard, averaged equally across the 10
teacher seeds. Negative values favor the optimized schedule. Intervals are
descriptive 95% seed-block bootstrap intervals; this is a post-hoc diagnostic,
not a new confirmatory experiment.

| NFE | Action RMSE difference (px) | Post-chunk keypoint RMSE difference (px) | Contact-only keypoint difference (px) | Absolute coverage-error difference |
| ---: | ---: | ---: | ---: | ---: |
| 2 | -0.877 | -0.420 `[-0.848, -0.148]` | -0.571 | -0.00279 |
| 3 | -0.491 | -0.145 `[-0.205, -0.094]` | -0.193 | -0.00158 |
| 4 | -0.480 | -0.123 `[-0.165, -0.086]` | -0.173 | -0.00125 |
| 5 | -0.360 | -0.093 `[-0.126, -0.066]` | -0.130 | -0.00087 |

The optimized schedule's lower one-chunk keypoint divergence is consistent
across all budgets, and each seed-block interval excludes zero. The effect also
strengthens rather than disappears on teacher-contact chunks.

## Interpretation

Experiment 2's failure to transfer is not explained by immediate physical
sensitivity along DDIM-10 states. The evidence now forms a three-level proxy
ladder:

1. **Offline action fidelity:** optimized schedules win at NFE 2--5.
2. **One-chunk physical fidelity:** optimized schedules still win at NFE 2--5.
3. **Complete closed-loop success:** optimized NFE 5 loses by 14 points, and
   the pooled method fails the non-inferiority gate.

Therefore the missing behavior occurs beyond a single local branch. Plausible
mechanisms are multi-query error compounding, recovery differences after the
policy visits its own off-teacher states, or sensitivity to the distribution
of states induced by the schedule. This diagnostic cannot distinguish those
mechanisms.

Do not optimize schedules against this one-chunk proxy: it would choose the
already-failed NFE-5 schedule again. Any further method needs an explicitly
multi-query or on-policy calibration design and a fresh final evaluation set.

