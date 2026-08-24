# Phase 1.5 matched-NFE result

## Decision

**Stop the adaptive residual-caching acceleration thesis.** Plain reduced-step
DDIM preserves the DDIM-10 action trajectory better at every tested compute
budget. Adaptive caching wins zero of the four budgets, below the precommitted
requirement of at least three.

This is a useful negative result: for this LeRobot PushT policy, spending a
fixed denoiser-call budget on a shorter native DDIM schedule is better than
running all ten scheduler steps while reusing residual transformations.

## Protocol

- Policy: `lerobot/diffusion_pusht_keypoints`
- Checkpoint revision: `58570fc39828d28efa5457aa297a52be27ac3a10`
- Dataset: `lerobot/pusht_keypoints`
- Dataset revision: `ace8c161a68bc025c21a5f29f85b86a9a2c5e64b`
- Reference: DDIM-10
- Samples: 100 real normalized two-frame conditioning examples
- Calibration/evaluation split: 25/75
- Evaluation strata: 30 early, 19 middle, 26 late temporal samples
- Budgets: 5, 6, 7, and 8 exact denoiser function evaluations
- Arms: DDIM-k, fixed residual reuse, and adaptive residual reuse
- Pairing: identical observation, initial noise, scheduler seed, and reference
- Integrity: 900 evaluation rows; every row's actual NFE equals its budget

Thresholds were selected from calibration traces using NFE only, never held-out
action error. The selected adaptive thresholds were 0.05 at budget 5 and 0.03
at budgets 6, 7, and 8.

## Results

Mean first-executed-action error against DDIM-10, converted to PushT pixels:

| NFE | DDIM-k | Fixed reuse | Adaptive reuse | Adaptive / DDIM |
| ---: | ---: | ---: | ---: | ---: |
| 5 | **0.316** | 2.330 | 3.833 | 12.11x |
| 6 | **0.260** | 0.885 | 1.505 | 5.79x |
| 7 | **0.157** | 0.507 | 0.609 | 3.88x |
| 8 | **0.137** | 0.251 | 0.275 | 2.01x |

Mean normalized full action-chunk MSE:

| NFE | DDIM-k | Fixed reuse | Adaptive reuse |
| ---: | ---: | ---: | ---: |
| 5 | **5.46e-6** | 1.55e-4 | 4.88e-4 |
| 6 | **3.08e-6** | 1.94e-5 | 6.86e-5 |
| 7 | **1.25e-6** | 5.88e-6 | 1.24e-5 |
| 8 | **7.43e-7** | 1.46e-6 | 2.51e-6 |

## Interpretation and limits

The result rejects this particular transformation-reuse rule on this policy;
it does not show that every caching method fails on every diffusion policy.
Residual reuse changes the scheduler trajectory while DDIM-k uses a schedule
designed for its shorter horizon, and the latter is consistently more faithful
here.

This experiment measures offline action fidelity, not closed-loop success. It
does not justify claims about CUDA latency, visual encoder cost, or task-success
non-inferiority. Those expensive follow-ups are intentionally skipped because
the primary matched-budget gate already failed.
