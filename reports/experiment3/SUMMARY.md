# Experiment 3: paired closed-loop PushT evaluation

## Outcome

**Decision: inconclusive-or-negative.**

The schedules selected by Experiment 2 were not shown superior or non-inferior
to standard DDIM-k in 400 frozen closed-loop episodes. Pooled success was 70.5%
for optimized schedules and 72.0% for standard schedules, a paired difference
of -1.5 percentage points. The two-sided seed-block bootstrap interval was
`[-10, +8]` points and the one-sided lower bound was -9 points.

## Results

| NFE | Standard successes | Optimized successes | Difference |
| ---: | ---: | ---: | ---: |
| 2 | 35/50 (70%) | 35/50 (70%) | 0 points |
| 3 | 34/50 (68%) | 37/50 (74%) | +6 points |
| 4 | 36/50 (72%) | 37/50 (74%) | +2 points |
| 5 | 39/50 (78%) | 32/50 (64%) | **-14 points** |

The NFE-5 point estimate violates the frozen rule that no individual budget
may fall more than five points below standard. The pooled one-sided lower bound
also fails the required `>-5`-point non-inferiority margin.

## Interpretation

Experiment 2 established that calibration-selected schedules approximate
DDIM-10 actions better offline. Experiment 3 shows that this advantage is not a
reliable substitute for closed-loop evaluation. In particular, the largest
offline action-fidelity gain occurred at NFE 5, where optimized closed-loop
success was worst relative to standard.

This does not reject few-step Diffusion Policy: standard schedules achieved
68--78% success with only two to five denoiser calls per query. It rejects the
claim that the current offline schedule-selection objective safely improves
task performance across budgets.

The full generated report is stored locally at
`outputs/experiment3/closed-loop/report.json` and remains gitignored.

