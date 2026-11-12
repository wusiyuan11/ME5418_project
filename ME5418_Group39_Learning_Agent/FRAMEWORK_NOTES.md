# Design rationale, stage boundary and open problems

## Main design advance to look for

The second submission already tracked the count of ordered arrivals in the environment, but did not give that counter to its policy. The third submission **exposes completed-waypoint history as a normalized fraction c/N** and **constructs the relative position of the next outstanding waypoint[c]**. For the submitted goal mode these are observation indices 21 and 18:21 (zero-based). This is a change to the actor/critic observation, not an increase in route points or a new/looser arrival test. If B was missed after the moving reference reaches C, the next-goal input still points to B.

This revision adds explanatory comments/docstrings and three observation/architecture regressions, for 36 total tests. Executable training syntax, task settings, weights and experiment data are unchanged from the previous verified build. The report now emphasizes the before/after design and separates full-task completion, ordered arrivals, timed-reference RMSE and original evaluation reward. Clear tables/figures are under `results/`; `CODE_GUIDE.md` maps features to source.

The information deficiency is addressed, but the reference remains time-driven, and hidden disturbances remain unobserved. Training success is evidence about the combined PPO-only system, not proof that the new inputs alone caused the gain. No new long training was performed for this presentation/comment revision.

## What the previous submitted report actually says

The Oct-30 Neural Network report describes custom PPO/GAE, fixed physical input scales, a tanh-squashed Gaussian and an inherited inner-loop controller. It explicitly says learning uses delayed task outcomes rather than supervised correct-command labels. The accompanying PPO source contains a clipped policy surrogate plus value MSE, with no behavior-cloning term. It reports 128 transitions and one update, not a trained mission policy.

It already discloses deviations from the proposal: direct PyBullet instead of gym-pybullet-drones; a 12 x 8 m rather than illustrative 20 x 20 m field; normalized hover-thrust deviation rather than direct total-thrust input; corrected actual ordered-arrival rewards. It also identifies the missing-next-waypoint/progress issue and suggests adding mission inputs as a future extension. These are implementation choices within RL, not evidence of imitation-assisted PPO. We cannot determine an individual's authorship from a group report.

A separate later local experiment DID use imitation/DAgger plus PPO. It is not that Oct-30 submission and is not this submission's main route. It must not be relabelled as pure PPO.

## What is new in the third milestone

1. Persistent full training instead of a one-update demonstration: eight independent physics clients, on-policy rollout, separate GAE per client, actor/value optimization, numerical guards, checkpointing and validation.
2. Public task observation 15 -> 23: the added velocity error, next unvisited goal, ordered progress and remaining time let the policy distinguish tasks with similar instantaneous tracking states but different waypoint history. These inputs do not estimate or expose true wind/liquid. This extends, rather than keeps identical, the previous network input interface.
3. A training-only reward/curriculum wrapper. Original environment sources and all evaluation success rules remain unchanged; the TRAINING objective and reset/termination distribution deliberately change.
4. Two-stage reinforcement learning: stabilization/coarse tracking PPO followed by goal-aware PPO continuation. No expert trajectory, teacher action, supervised action regression or imitation checkpoint initialization.
5. Reproducible main config/entry point, exact dependencies, tests, seed-disjoint model selection, raw training/evaluation logs, actual policy replay and a two-page shared report.

## Exact mission training feedback

At every 0.05 s physics step in the submitted mission configuration:

```text
r_train = 0.1 * r_original
          + 0.3 * (Phi_after - Phi_before)
          + 10 * actual_ordered_waypoint_event
          + 30 * success - 30 * timeout
          - 0.1 * min(12, distance_to_next_goal) * dt
          - 0.01 * yaw_angle_squared * dt
Phi = cumulative_path_length_at_next_goal - distance_to_next_goal
```

The raw reward still contains tracking, reference-velocity, angular-rate and action-effort penalties, ordered waypoint bonuses and crash/boundary penalties. A held decision sums its actual microstep rewards; discount/GAE run at the decision interval. The progress difference is a **heuristic shaped objective**, not a claim of policy-invariant potential-based reward shaping: no such invariance proof is made. The reference and next-goal objectives can still conflict after a miss. The goal-distance term is bounded to avoid uncontrolled penalty scale, not to relax arrival tolerance.

Early training: wind limit 0.5 -> 3 m/s by half of the decision budget; some resets use positions/velocities from the public desired path, with probability 0.8 -> 0 by three quarters. Artificially skipped prefix waypoints are logged separately and never rewarded. An optional missed-waypoint cutoff at scheduled arrival time + 1.5 s focuses attempts early in training; it is removed at three quarters. The final quarter uses the original start-at-rest distribution, full wind and original deadline. A best checkpoint may be selected earlier, but ALL validation and held-out tests use the original full mission without any training assistance.

PPO defaults: 8 clients x 256 decisions per rollout = 2048 transitions; minibatch 256, 8 epochs, policy lr 3e-4, value lr 1e-3, gamma .9975 per 0.1 s decision, GAE lambda .97, ratio clip .2, gradient norm .5, KL threshold .02 (actor stops when minibatch estimate > 1.5 times threshold), linear lr decay toward 10%. Actor/critic have separate parameters and Adam. Critic MSE uses interaction-derived return targets; it is not imitation learning.

## Why these changes, and will they work well?

The older policy could reduce timed-reference RMSE yet miss an ordered turn: the clock kept moving while its required waypoint stayed behind. Making the outstanding goal observable and rewarding actual completion makes the objective more explainable. Action hold makes exploration temporally correlated at a 10 Hz outer loop while physics/inner control remain at 20 Hz. Curriculum and PPO-only staging improve access to difficult turns without supplying a correct action.

These are testable hypotheses, not guaranteed improvements. Early observation-only/weak-reward/curriculum trials failed; a later combined configuration succeeded. We have not causally isolated each component. The published seed-7 main model completed 50/50 held-out episodes, but a second continuation seed scored 36/50, direct random initialization scored 0/50, and stronger classical control had better tracking. The two positive continuation branches share the same foundation and are NOT two independent end-to-end training seeds.

## Next steps for the final project

- Improve independent from-scratch convergence and add multiple independently trained foundations; report uncertainty across complete training seeds.
- Run single-component ablations with matched physical-step budgets, task information and control rates. Use fresh held-out seeds for future model selection.
- Resolve reference-vs-arrival scheduling more systematically, reduce corner overshoot and altitude variation without sacrificing completion; assess energy/action-change terms rather than calling squared command physical energy.
- Compare observation stacking/GRU for hidden disturbances; randomize routes and speeds, and complete nominal/wind-only/payload-only/combined/unseen-gust comparisons against a strong tuned PID baseline.
- Improve simulator fidelity and assess continuous spray coverage/deposition and gust recovery. Sampled waypoint proximity is not spray-uniformity evidence.
- Add resumable new-stage vector training if long unattended runs require it; currently live physics states are not checkpointed.

## Libraries, external code and AI assistance

PyTorch provides neural layers/autograd/Normal/Adam, NumPy calculations, Gymnasium API/spaces, PyBullet rigid-body/contact physics, Matplotlib/Pillow visuals, pytest tests and PyYAML dependency verification. Network, PPO/GAE, task wrapper, curriculum, rollout, metrics and integration are group implementation extending the previous submission. No SB3, gym-pybullet-drones, external policy weights or expert datasets are used. Codex assisted code, experiments, testing, packaging and writing.

Method/library documentation:

- Schulman et al. (2017), https://arxiv.org/abs/1707.06347
- OpenAI Spinning Up, https://spinningup.openai.com/en/latest/algorithms/ppo.html
- PyTorch 2.2 distributions, https://docs.pytorch.org/docs/2.2/distributions.html
- Gymnasium API, https://gymnasium.farama.org/api/env/
- PyBullet Quickstart, https://github.com/bulletphysics/bullet3/tree/master/docs/pybullet_quickstart_guide
