# Experiment 2D protocol: one-chunk rollout sensitivity

Experiment 2D is a post-hoc diagnostic extension of Experiment 2. Experiment 3
remains the frozen 400-episode closed-loop evaluation. This diagnostic cannot
change either experiment's decision or support a new superiority claim.

## Question

Does one-chunk physical divergence from DDIM-10 explain why Experiment 2's
offline action-fidelity gains transferred inconsistently in Experiment 3?

## Counterfactual design

- Run the pinned DDIM-10 policy as a teacher in PushT.
- At every eight-action policy-query boundary, capture the teacher's two-frame
  conditioning history and exact simulator body state, including velocities.
- From that same state and the same deterministic initial diffusion noise,
  generate the standard and Experiment 2 optimized action chunks for NFE
  budgets 2, 3, 4, and 5.
- Restore the snapshot before every arm and execute exactly eight open-loop
  actions. The teacher trajectory then continues independently to provide
  later query states.

This design measures local physical sensitivity without retuning schedules or
rerunning the Experiment 3 decision.

## Metrics

For every teacher snapshot and method, record:

- action-chunk RMSE against DDIM-10 in PushT pixels;
- final block-keypoint and agent-position RMSE against DDIM-10 in pixels;
- signed and absolute final coverage difference from DDIM-10;
- teacher and candidate contact counts during the chunk; and
- whether the teacher chunk contains contact.

Summaries report each metric overall and separately for teacher-contact chunks,
plus optimized-minus-standard paired means at each NFE budget.

## Interpretation

This is exploratory and post-hoc. A useful rollout proxy should at minimum rank
the optimized NFE-3 schedule as no more divergent than standard and the
optimized NFE-5 schedule as more divergent than standard, matching the signs
of Experiment 3's observed success differences. Failure to do so stops this
proxy before any rollout-aware schedule search. Success only authorizes a
separately specified calibration search on fresh seeds; it is not evidence
that such a search will improve task success.

