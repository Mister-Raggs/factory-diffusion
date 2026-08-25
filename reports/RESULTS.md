# Factory Diffusion results

This repository preserves a sequence of frozen, matched-compute experiments on
inference shortcuts for LeRobot's keypoint-conditioned PushT Diffusion Policy.
Negative results are retained alongside the implementation and protocols.

| Stage | Question | Evidence | Decision |
| :--- | :--- | :--- | :--- |
| Phase 1.5 | Does adaptive residual reuse beat simply taking fewer DDIM steps at matched NFE? | 75 held-out observations, four NFE budgets | **Stop caching:** DDIM-k won 4/4 budgets |
| Experiment 2 | Can calibration-selected timesteps better reproduce DDIM-10 actions? | 25 calibration + 75 held-out observations | **Offline positive:** optimized won 4/4 budgets |
| Experiment 3 | Do those schedules preserve closed-loop success? | 400 paired PushT episodes | **Failed non-inferiority:** pooled -1.5 points; NFE 5 -14 points |
| Experiment 2D | Does one-chunk physical divergence explain the failure? | 244 teacher states, 1,952 branches, 169 contact states | **Stop proxy:** it still favored the failed NFE-5 schedule |

## Defensible result

> Offline action fidelity—and even one-chunk physical fidelity—was not a
> reliable proxy for closed-loop success when selecting few-step Diffusion
> Policy schedules.

The result is intentionally narrow. It applies to the pinned PushT keypoint
policy, the tested two-to-five-step DDIM schedules, and the frozen protocols in
this repository. It does not show that schedule optimization always hurts,
that the optimized schedules are statistically worse overall, or that the
finding generalizes to other policies and tasks.

## Durable artifacts

- [Phase 1.5 matched-NFE result](phase15/SUMMARY.md)
- [Experiment 2 offline schedule result](experiment2/SUMMARY.md)
- [Experiment 2D physical-sensitivity diagnostic](experiment2/ROLLOUT_SENSITIVITY.md)
- [Experiment 3 closed-loop result](experiment3/SUMMARY.md)
- Machine-readable summaries live beside each report as JSON.

Raw checkpoints, datasets, row-level reports, traces, and videos are excluded
from git. Every committed summary records enough immutable identifiers and
protocol information to locate the corresponding runner and reproduce it.

