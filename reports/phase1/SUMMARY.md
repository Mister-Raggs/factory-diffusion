# Phase 1 preliminary result

Decision: **proceed-to-cuda-and-closed-loop-validation** (provisional).

> **Superseded after hardening review:** Phase 1 is now treated as a mechanism
> smoke test because it used synthetic conditioning and omitted a matched-NFE
> DDIM baseline. The active decision is to run zero-credit Phase 1.5 before
> CUDA or closed-loop work. See `docs/PROJECT_CONTEXT.md`.

Checkpoint: `lerobot/diffusion_pusht_keypoints` at `58570fc39828d28efa5457aa297a52be27ac3a10`.
Seeds: `[0, 1, 2, 3, 4, 5]`. Device: `cpu`.
Mean denoiser share: `0.998`.

| Threshold | Mean skip | Worst first-action error | Mean chunk MSE | Median CPU speedup* |
| ---: | ---: | ---: | ---: | ---: |
| 0.0000 | 0.0% | 0.000000 | 0 | 0.99x |
| 0.0750 | 13.3% | 0.005529 | 1.99167e-05 | 1.10x |
| 0.1000 | 21.7% | 0.072094 | 0.000219658 | 1.32x |
| 0.1500 | 30.0% | 0.033860 | 0.000178404 | 1.45x |
| 0.2000 | 35.0% | 0.040214 | 0.000228936 | 1.43x |
| 0.2500 | 35.0% | 0.043877 | 0.000382524 | 1.53x |

\* Functional local timing only; not a publishable CUDA claim.

This gate only establishes that adaptive reuse has measurable skip opportunities with bounded normalized action deviation. Task success and CUDA latency remain Phase 2.
