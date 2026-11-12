# Code reading guide - the third-milestone improvement

## 1. Start with actual waypoint progress, not reference-time progress

Read `pure_rl/task.py: mission_observation`. The old environment already counts ordered physical arrivals in `waypoints_achieved`. The new work is feeding that history to the policy and critic:

| Observation slice (zero-based) | Meaning | Why it matters |
|---|---|---|
| 0:15 | Original motion and timed-reference error | Retains the previous interface as the first features |
| 15:18 | Reference velocity minus actual velocity in the submitted goal preset | Exposes the commanded motion, not just current velocity |
| 18:21 | waypoint[c] minus current position | A missed point remains the outstanding goal even when reference time advances |
| 21 | c/N, ordered-arrival fraction | Makes task completion history visible |
| 22 | Remaining time / 45 s | Distinguishes stages of the finite-deadline mission |

At full-task reset, the start is counted once without reward. Only the next ordered point is eligible for the original 0.5 m 3-D arrival check. The new policy receives a summary of the declared task history; it does not receive hidden wind, liquid mass, a PID action or the Critic's output.

Counterfactual example: at C, one vehicle has previously visited B and another missed B. Their original 15 values can be identical, but their required next point is different. `tests/test_mission_progress.py` constructs this input-contract check; it is not a fabricated successful rollout.

## 2. Follow the reward and curriculum boundary

`pure_rl/task.py: LearningTask._physics_step` keeps the base reward for logging and uses training-only feedback. Arrival bonus uses **new count minus previous count**, not the absolute count or the clock's route progress. Curriculum resets declare skipped prefixes separately; they earn no reset bonus and do not become full-task evaluation evidence. Action holds still check physical arrivals/safety on each 0.05 s step.

Reference generation is NOT changed: it still follows elapsed time. The new inputs expose a missed obligation, rather than eliminating every reference-versus-arrival conflict.

## 3. Separate Actor decisions from Critic training support

`pure_rl/agent.py` extends the inherited architecture to Actor 23-64-64-4 and Critic 23-64-64-1. The two networks see the same 23 values, but have separate weights. Critic output does not enter Actor input. `pure_rl/learning.py` uses interaction-derived advantages for PPO policy loss and return targets for value MSE; there is no expert-action regression. Comments identify how independent optimizers and KL stopping work.

## 4. Read sampling before the optimizer

`pure_rl/collector.py` preserves unfinished episodes across updates, captures final values before reset, computes GAE down time for each client, then flattens the batch. It separately logs raw/shaped reward, decision/physics counts, declared curriculum prefixes and newly earned arrivals. Comments explain why mixing these would misstate progress.

## 5. Keep evaluation independent

`pure_rl/train.py: evaluate` uses the original environment directly, with the same public feature construction/control rate but no training wrapper. `selection_key` prioritizes completion, then ordered points, safety and tracking error. `pure_rl/evaluate.py` checks seed overlap and writes all episode metrics; a replay is one real example, not an aggregate score.

`run_learning_agent.py` and `configs/mission_ppo.json` are the main entry and settings. See `README.md` for exact commands and checkpoint-format/restart limits. The original environment and `rl/network.py` remain byte-identical to the previous course submission; newly added comments are in the third-milestone layer.

## 6. Understand the result labels

The previous course submission had only a 128-transition update demonstration. Later trained 15D baselines are not that submitted smoke checkpoint. The current selected 23D model has PPO-only staged training. Compare it with both negative seeds/random starts and tuned PID; read `results/RESULTS_SUMMARY.md` and source CSVs. The new observation is an explicit design advance; its isolated causal contribution remains unproven.
