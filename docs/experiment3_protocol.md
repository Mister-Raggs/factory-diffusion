# Experiment 3 protocol: paired closed-loop PushT

This protocol is frozen before any closed-loop policy outcome is inspected.

## Question

Do the calibration-selected schedules from Experiment 2 preserve closed-loop
PushT task success at exact NFE budgets `k in {2, 3, 4, 5}` better than, or at
least non-inferior to, the standard DDIM-k schedules?

## Fixed methods

The checkpoint, normalization statistics, and schedules remain unchanged from
Experiment 2. There is no closed-loop schedule tuning.

| NFE | Standard | Optimized |
| ---: | :--- | :--- |
| 2 | `(50, 0)` | `(70, 0)` |
| 3 | `(66, 33, 0)` | `(80, 10, 0)` |
| 4 | `(75, 50, 25, 0)` | `(90, 50, 10, 0)` |
| 5 | `(80, 60, 40, 20, 0)` | `(90, 70, 30, 10, 0)` |

Every policy query executes exactly the schedule length in denoiser calls. The
policy executes eight actions from each predicted chunk, matching the pinned
LeRobot checkpoint configuration.

## Environment and pairing

- Environment: `gym-pusht>=0.1.5,<0.2.0`, keypoint observation mode
  `environment_state_agent_pos`.
- Maximum episode length: 300 environment steps.
- Evaluation seeds: `0..49` for every method and budget.
- Integration-only smoke seeds: `1000` and `1001`; smoke outcomes may not
  change schedules, metrics, seeds, or the decision rule.
- A pair uses the same environment seed and the same deterministic initial
  diffusion noise for every corresponding policy-query index.
- Observation normalization and action unnormalization use the pinned
  `lerobot/pusht_keypoints` dataset statistics.
- Actions are clipped to the environment action bounds after unnormalization;
  clipping frequency is reported.
- Rendering is disabled during the scientific run. Videos are a later
  presentation artifact and not an experimental input.

Trajectories may diverge after the methods choose different actions. Common
random numbers preserve pairing but do not imply identical later observations.

## Outcomes

The primary per-episode outcome is success: whether coverage reaches the
environment's 95% threshold at any point. Report paired success counts and the
optimized-minus-standard success-rate difference for each NFE and pooled
across NFE.

Secondary outcomes are maximum coverage, sum reward, steps to success for
successful episodes, policy-query count, exact total NFE, and action clipping
frequency.

Use a seed-block bootstrap for pooled confidence intervals: resample the 50
environment seeds and keep all four NFE budgets for each sampled seed. Report
a one-sided 95% lower confidence bound and a two-sided 95% interval for the
pooled success-rate difference. Per-budget intervals are paired bootstraps over
the 50 seeds. Report discordant success pairs without relying on Gymnasium's
`final_info` aggregation.

## Decision rule

The optimized schedules are closed-loop non-inferior if:

1. the one-sided 95% seed-block-bootstrap lower bound for the pooled paired
   success-rate difference is greater than `-0.05`; and
2. no individual budget has an optimized-minus-standard success-rate point
   estimate below `-0.05`.

Claim superiority only if the two-sided 95% pooled interval lies entirely
above zero. Otherwise report non-inferiority, inconclusive evidence, or a
negative result exactly as observed.

Only a non-inferior result authorizes a separately frozen CUDA wall-clock
benchmark. Factory SRE integration remains optional and follows the standard
PushT result.
