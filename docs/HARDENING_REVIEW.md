# Factory Diffusion — Hardening Review

**Reviewed:** 2026-08-23
**Subject:** `Mister-Raggs/factory-diffusion` @ `e61a6b3` (Phase 1 complete)
**Basis:** Phase 1 summary + repo source + external prior-art search
**Web version:** https://claude.ai/code/artifact/78617b79-5863-4e9c-a1a1-846cda7c1cb2

---

## Verdict

| Dimension | Assessment |
| --- | --- |
| **Mechanism** | **Sound.** Cache resets per chunk, threshold 0 is exact, warmup / final-step / max-skip guards are correct. The code is not the problem. |
| **Framing** | **Not defensible.** Claiming a 1.45x win over a baseline whose free competitor — fewer DDIM steps — has never been run. |
| **Downstream (Factory SRE)** | **Overbuilt.** Most expensive phase, least evidential. Demote from experiment to demo. |

**One paragraph:** Pursue it, but not as currently framed. "Training-free caching makes Diffusion Policy faster" is a crowded claim you are positioned to lose — direct prior art exists, distillation beats you by an order of magnitude, and your own baseline choice already banked the easy 10x. What is *not* taken, and what your harness is unusually well built to answer, is the question the whole caching literature quietly skips: **at a matched compute budget, does adaptive caching actually beat just taking fewer denoising steps?** That reframe costs nothing, is answerable this week on your laptop, and is publishable whichever way it lands.

---

## Findings, severity ranked

### S1 — BLOCKING: fighting for 1.45x on top of a 10x you already took

Your baseline is `DDIM` with `num_inference_steps=10` (`src/factory_diffusion/baselines/pusht_keypoints.py:38-39`). That is already the strong, fast configuration. The published "Diffusion Policy is too slow" numbers (1.5 Hz, >100 ms) come from **100-step DDPM**.

So threshold 0.15 skipping 3 of 10 U-Net calls does not compete against a slow baseline. It competes against **running DDIM with 7 steps** — zero lines of code, identical NFE.

That comparison currently sits at item 7 of Phase 2, gated behind GPU credits. It needs neither. It is a pure action-error comparison runnable today, and it is the single most likely thing to kill the project.

**And the ceiling is lower than 1.45x.** The reported "denoiser = 99.8% of sampler time" comes from the *keypoint* policy, which has no vision encoder at all. On a visual policy — the case Factory SRE would actually use — published breakdowns put the encoder at roughly 40-60% of end-to-end latency. Under Amdahl, 1.45x on a U-Net that is 45% of the loop is **1.16x end-to-end**. That is the number a reviewer will hold you to.

### S2 — NOVELTY: prior art is closer than the README assumes

The README frames this as an untested transfer from video diffusion to robot policies. That transfer has already been made and published — including the specific observation that generic diffusion-caching methods don't carry over to Diffusion Policy.

| Work | What it does | Result | Overlap |
| --- | --- | --- | --- |
| **BAC** (arXiv:2506.13456) | Training-free block-wise adaptive caching *for Diffusion Policy*. PushT, Robomimic, Kitchen, BlockPush. | up to 3.4x | **Direct** |
| **VLA-Cache** (NeurIPS 2025) | Adaptive token caching across frames for VLA manipulation. | — | Adjacent |
| **ActionCache** (arXiv:2607.06370) | Reuses past intermediate actions to warm-start flow-matching action heads. | 10-40x | Adjacent |
| **OneDP / Consistency Policy** | Distills the denoiser to one step (requires training). | 1.5 -> 62 Hz | **Dominates** |
| **Real-Time Chunking** (arXiv:2506.07339) | Hides inference latency entirely via async chunk inpainting. Already in LeRobot. | — | **Undercuts motive** |

The differentiator as written — CNN U-Net rather than BAC's transformer backbone — is thin. A reviewer reads that as an ablation, not a contribution.

**But here is the actual gap.** BAC's baseline is **K = 100 denoising steps**, and it compares only against full precision and uniform caching. It never compares against simply running fewer DDIM steps. Neither does most of this literature. A 3.4x speedup measured down from 100 steps may be doing nothing that DDIM-10 does for free — and *nobody has checked*. You are one week of laptop compute from being the person who checked.

### S3 — VALIDITY: Phase 1's "proceed" rests on inputs the policy has never seen

In `experiments/01_pusht_keypoints_probe.py:80`, conditioning is `global_cond = uniform_(-1, 1)` over a 36-dim vector. That is not a PushT observation — real conditioning is two stacked frames of normalized keypoints living on a thin manifold. Denoising trajectories on out-of-distribution conditioning can be far smoother than real ones, which would inflate every skip fraction in the table. The limitation is noted in `summary.json` but the headline reads `proceed-to-cuda-and-closed-loop-validation`; that decision is not yet earned.

Two more things in the same table:

- **The threshold is not controlling the error it claims to control.** Worst first-action error at tau=0.10 is `0.0721` — larger than tau=0.15 (`0.0339`), tau=0.20 (`0.0402`) and tau=0.25 (`0.0439`). EasyCache's premise is that accumulated predicted error bounds real error. On n=6 it visibly doesn't. Either the estimator is mis-specified for this model, or six samples is pure noise. Both are worth knowing before Phase 2.
- **The error criterion has no task meaning.** PushT actions are pusher targets in a 512 px frame, min-max normalized to [-1, 1]. So `0.0339` ~= **8.7 pixels** on the very first executed action. Put that next to the pusher radius and the goal-coverage tolerance to learn whether the 0.05 gate was generous or absurd. Right now it is arbitrary.

---

## Is the result measurable?

Partly — but not in the shape the plan describes. Both problems are fixed by changing experiment design, not code.

### A latency-vs-success Pareto curve is not measurable in simulation

Sim doesn't run in real time. Making inference faster changes **nothing** about closed-loop success unless you explicitly couple wall-clock to the physics step. The curve you can actually produce is **compute versus action fidelity**, where success only ever goes down. You cannot demonstrate a benefit — only bound a cost.

**Reframe the axis:** *"at a fixed budget of N function evaluations, which method gives the highest success rate?"* That is measurable, honest, and it makes the DDIM-N comparison the centerpiece rather than an afterthought.

### Power: know your MDE before spending a GPU-hour

The official PushT checkpoint reports ~65.4% success over 500 episodes. Detecting a 5 pp degradation between two independent arms at 80% power needs roughly **1,460 episodes per arm**. Running 50 episodes and reporting "success unchanged" carries an honest CI of about +/-13 pp — wide enough to hide a catastrophic regression.

Your harness already does the right thing and should lean into it: **paired episodes under common random numbers** — same seed, same initial noise, arms differing only in skip decisions. Score with McNemar on discordant pairs. Pre-register the non-inferiority margin (e.g. "cached is non-inferior if the lower bound of the paired 95% CI is above -5 pp") and report the achieved MDE alongside every success number.

---

## Phase 1.5 — three experiments, zero credits, one week

Do not start Phase 2. Do not touch Factory SRE. These run on the machine you already have and either save the project or end it cheaply.

### A. Real conditioning — ~half a day

Replace the uniform random `global_cond` with real observations. Pull them from the public `lerobot/pusht` dataset on the Hub, or roll out `gym-pusht` locally. No training, no credits. Sample 100+ conditioning vectors spanning early, mid and contact-phase states — residual stability almost certainly differs near contact. Re-run the threshold sweep. If skip fractions collapse on real observations, that is your answer.

### B. The matched-NFE shootout — ~one day, decides the project

Fix a reference trajectory: DDIM with 10 steps, full compute. Then for each budget k in {5, 6, 7, 8}, compare three arms that all spend exactly k U-Net evaluations:

1. plain DDIM-k
2. fixed-interval skipping
3. your adaptive cache tuned to land on k

Measure action-chunk MSE and first-action error against the DDIM-10 reference, over the real conditioning set from A.

This is the experiment the entire literature is missing and the one the whole thesis rests on. **Run it first if you only run one.**

### C. The Amdahl check — ~two hours

Measure the encoder-vs-U-Net split on the *visual* policy, not the keypoint one. You don't need the broken `lerobot/diffusion_pusht` checkpoint — time a ResNet-18 forward at the policy's input resolution against 10 U-Net calls at the real batch shape. That gives the honest end-to-end speedup ceiling, and tells you whether the denoiser is even the interesting engineering target.

### Kill criteria — write these down before running

- If adaptive caching at budget k does **not** beat plain DDIM-k on action error for at least three of four k values -> the acceleration thesis is dead. Stop, and write it up as the negative result (which is worth writing).
- If skip fractions on real observations fall below ~20% at any threshold with first-action error inside a task-derived tolerance -> stop.
- If the encoder is >60% of visual-policy latency -> the denoiser is the wrong target; say so and stop.

---

## What to claim, and what to cut

Assuming Phase 1.5 survives, restructure the deliverable around the gap rather than the mechanism.

| Element | Current plan | Hardened plan |
| --- | --- | --- |
| **Title claim** | Adaptive residual caching accelerates Diffusion Policy | Do caching methods beat fewer denoising steps for diffusion policies? A matched-budget study |
| **Primary axis** | Latency vs. success Pareto | Fixed NFE budget vs. success, across DDIM-k / fixed-interval / adaptive |
| **Environments** | PushT, then Factory SRE battery docking | PushT + 2 Robomimic tasks. Standard, comparable, already in LeRobot. |
| **Factory SRE** | Phases 10-15: adapter, scripted demos, train a policy, evaluate | **Demote.** Motivation and one demo video. No experimental arm. |
| **Success bar** | "Success rate maintained" | Pre-registered non-inferiority margin + reported MDE on paired episodes |

**Why cut Factory SRE from the experiments.** Phases 11-15 are the most expensive work in the plan — build an adapter, script an expert, collect demos, train a policy — and they produce the weakest evidence. A caching result in a bespoke environment is not comparable to anything, and it is confounded by the prior question of whether the policy learned battery docking at all. If the policy underperforms, you cannot separate "caching hurt" from "imitation learning didn't take." Every scientific claim you want lives on the standard benchmarks. Keep Factory SRE as the reason the question matters and as a video at the end — genuinely valuable for a hackathon deliverable, at a fraction of the cost.

**One clarity note.** The repository is named for the downstream demo but is actually a diffusion-caching methods repo. Once Factory SRE is demoted, that mismatch will confuse every new reader. Either rename, or put one sentence at the very top of the README saying plainly what the repo is and what it is not.

---

## Compute budget — where credits are and aren't spent

**Phase 1.5 spends zero credits by design.** Experiments A, B and C are laptop-scale. Their entire purpose is to gate credit spend behind a result: you do not buy a GPU-hour until B says adaptive caching beats DDIM-k at matched budget.

*(Optional accelerant: experiment B is embarrassingly parallel over conditioning samples. ~2 GPU-hours turns it from a day into minutes. Not required — do not spend it if credits are tight.)*

**The revised Phase 2 does need credits.** Estimated below in GPU-hours, since I don't know how Antioch / Token Factory credits are denominated — map these to your own units before committing.

| Item | Credits? | Est. GPU-hr |
| --- | --- | --- |
| Phase 1.5 (A/B/C) | No | 0 |
| CUDA latency profiling: warm/cold, sync'd, `torch.compile` baseline | Yes | 2-4 |
| PushT closed-loop: 13 arms x 500 paired episodes | Yes | 5-10 |
| Robomimic task 1 (image obs, MuJoCo) | Yes | 15-25 |
| Robomimic task 2 | Yes | 15-25 |
| Reruns, debugging, failed sweeps (~1.5x contingency) | Yes | ~30 |
| **Subtotal** | | **~70-100** |
| *Contingency:* training DP on Robomimic if released checkpoints don't load | Maybe | +24-48 **per task** |

**13 arms** = 4 budgets (k in {5,6,7,8}) x 3 methods (DDIM-k / fixed-interval / adaptive) + 1 full-compute reference.

### Three things that drive this number

1. **Checkpoint availability is the big swing.** If the original `diffusion_policy` repo's released Robomimic checkpoints load into your harness, Phase 2 is ~70-100 GPU-hr. If they don't and you have to train, each task is another 1-2 GPU-days and the budget roughly doubles. **Check checkpoint loading during Phase 1.5** — it costs nothing and it's the difference between a cheap Phase 2 and an expensive one. If both fail, drop to PushT plus one cheap state-based task rather than paying to train.
2. **Sim stepping is often the bottleneck, not the GPU.** PushT (pymunk) and Robomimic (MuJoCo) step on CPU. Vectorized envs and enough CPU workers matter more than GPU class here. Budget CPU alongside credits.
3. **Calibrate before committing.** Run one arm x 20 episodes, measure wall-clock, multiply by 13 arms x 25. Do that before authorizing the full sweep — my estimates are order-of-magnitude and your actual per-episode cost is the only number that matters.

### What the cut saves

Demoting Factory SRE from experimental arm to demo removes scripted demo generation, DP training on a bespoke task, and closed-loop eval through Antioch — plausibly another 50-100+ GPU-hours and most of the calendar time, for evidence that wasn't comparable to anything anyway. **The hardened plan is cheaper than the original, not more expensive**, and it spends what it does spend on results that stand on their own.

---

## Sources

1. [Block-wise Adaptive Caching for Accelerating Diffusion Policy](https://arxiv.org/abs/2506.13456) — arXiv:2506.13456
2. [Less is Enough: Training-Free Video Diffusion Acceleration via Runtime-Adaptive Caching (EasyCache)](https://arxiv.org/abs/2507.02860) — arXiv:2507.02860
3. [VLA-Cache: Efficient Vision-Language-Action Manipulation via Adaptive Token Caching](https://arxiv.org/abs/2502.02175) — NeurIPS 2025
4. [Training-Free Acceleration for VLA Models with Action Caching and Refinement](https://arxiv.org/abs/2607.06370) — arXiv:2607.06370
5. [One-Step Diffusion Policy: Fast Visuomotor Policies via Diffusion Distillation](https://arxiv.org/abs/2410.21257) — arXiv:2410.21257
6. [Consistency Policy: Accelerated Visuomotor Policies via Consistency Distillation](https://consistency-policy.github.io/)
7. [Real-Time Execution of Action Chunking Flow Policies (RTC)](https://arxiv.org/abs/2506.07339) — arXiv:2506.07339 · [LeRobot RTC docs](https://huggingface.co/docs/lerobot/rtc)
8. [lerobot/diffusion_pusht model card](https://huggingface.co/lerobot/diffusion_pusht/blob/main/README.md) — reported PushT success rate
