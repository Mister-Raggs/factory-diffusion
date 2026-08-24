# Experiment 2: task-calibrated DDIM schedules

## Outcome

**Decision: proceed to a paired closed-loop PushT evaluation.**

A schedule selected using only 25 real PushT calibration observations improved
offline DDIM-10 action fidelity over the standard DDIM-k schedule at every
exact denoiser-call budget from two through five. This passes the frozen
three-of-four gate with four wins.

## Frozen protocol

- Policy: `lerobot/diffusion_pusht_keypoints`
- Checkpoint revision: `58570fc39828d28efa5457aa297a52be27ac3a10`
- Dataset: `lerobot/pusht_keypoints`
- Dataset revision: `ace8c161a68bc025c21a5f29f85b86a9a2c5e64b`
- Seed: 0
- Samples: 100 total; samples 0--24 calibration, 25--99 held out
- Inference batch size: 3
- Reference: full DDIM-10 schedule `(90,80,70,60,50,40,30,20,10,0)`
- Candidates: every exact-k descending schedule ending at zero on
  `{0,10,...,90}`, plus the standard schedule when absent
- Selection: calibration mean normalized action-chunk MSE only
- Gate: lower held-out mean chunk MSE and no higher mean first-action error at
  three of four budgets

Candidate counts were 9, 37, 85, and 126 for NFE 2, 3, 4, and 5.

## Held-out results

| NFE | Standard | Optimized | First-action px: standard | Optimized | Chunk MSE: standard | Optimized |
| ---: | :--- | :--- | ---: | ---: | ---: | ---: |
| 2 | `(50,0)` | `(70,0)` | 3.1065 | **1.4474** | 3.4614e-4 | **1.0584e-4** |
| 3 | `(66,33,0)` | `(80,10,0)` | 0.8366 | **0.3603** | 6.6198e-5 | **7.0394e-6** |
| 4 | `(75,50,25,0)` | `(90,50,10,0)` | 0.4932 | **0.1095** | 1.6550e-5 | **1.1472e-6** |
| 5 | `(80,60,40,20,0)` | `(90,70,30,10,0)` | 0.3164 | **0.0490** | 5.4602e-6 | **2.3258e-7** |

Each cell summarizes the same 75 held-out observations and initial noise. All
paired 95% bootstrap confidence intervals for optimized-minus-standard first-
action error and action-chunk MSE lie entirely below zero.

## Interpretation

The useful finding is narrow and clear: for this pretrained Diffusion Policy,
*which* denoising timesteps are used matters substantially at very small NFE.
A tiny task-specific calibration split found schedules that approximate the
full DDIM-10 action chunk more accurately without retraining or adding model
calls.

This does not yet establish improved PushT task success. It is also not a
wall-clock speed claim: both methods use equal exact NFE, and CPU timings are
diagnostic. Closed-loop evaluation is the necessary next experiment.

The complete generated report, including 600 paired held-out rows and all 257
calibration candidates, is stored locally at
`outputs/experiment2/schedules/report.json` and remains gitignored.
