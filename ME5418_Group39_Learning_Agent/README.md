# ME5418 Group 39 - Learning Agent C&R

This is the third milestone's learning-agent source and recorded results. The shared **two-page report** is `report/ME5418_Group39_Learning_Agent_Report.pdf`. The emphasis is a reasonable, explainable, trainable framework, not a claim that RL must beat PID.

This directory is the stable GitHub location, synchronized from the verified 6 November course materials on 12 November 2026. Training code, configuration, models and measured values are retained; repeated evidence is stored once and host-specific packaging logs are omitted. No new long training was performed for this synchronization. The separate course ZIP is not committed, to avoid duplicating the whole source tree.

This updated submission adds explanatory code comments and three focused regression tests, highlights the waypoint-aware design advance in the two-page report, and separates completion/error/reward results more clearly. Training logic, settings, model weights and measured experiment data are unchanged; comments/docstrings are verified separately from executable syntax. The plain report style follows the previous milestone reports. Its editable text is `report/REPORT_SOURCE.md`; `本次提交说明.md` is a short Chinese handoff explanation.

## Main advance since the previous Neural Network submission

The old environment already recorded ordered arrivals, but its 15-input policy could not observe that history. The new policy receives **completed-waypoint fraction c/N** and **position error to the next outstanding waypoint[c]**, plus reference-velocity error and remaining time. When reference time moves past a missed point, the outstanding goal remains visible. We add actual-arrival-oriented training feedback and a sustained PPO workflow; we do not add new route points or relax success. See `CODE_GUIDE.md` for feature indices and annotated code locations.

The previous course submission demonstrated 128 transitions and one PPO update, not a trained success rate. The trained 15D baselines in our results were produced afterwards. Changes were explored together, and observation-only trials failed; do not attribute the entire success gain to the new inputs alone. Reference itself still advances by time.

## Read the results first

- `results/RESULTS_SUMMARY.md`: original-task completion, ordered arrivals, tracking RMSE, evaluation return, validation learning progress and stress results, with their limitations.
- `results/heldout_comparison.png`: completion and tracking shown side by side; the same successful policy is not necessarily the most accurate tracker.
- `results/training_rewards.png`: shaped training reward and original raw reward shown separately, using recorded full-start training attempts.
- `results/comparison.csv`: machine-readable controller means. Rebuild these presentations with `python -m tools.summarize_results`; this performs no training or new model selection.

The agent is **custom PyTorch PPO**, extending the group's Oct-30 environment and actor-critic. It is not Stable-Baselines3, behavior cloning or PID-guided action learning. No teacher data or imitation model is loaded by default. The successful supplied checkpoint has a disclosed **two-stage PPO-only** training history. Random-start runs can update successfully without completing the mission.

## 1. Install and verify

From this directory, with Conda installed:

```bash
conda env create -f environment.yml
conda activate me5418-learning-agent
python verify_setup.py --strict-versions
python -m pip check
python -m pytest -q
python -m tools.verify_submission
```

`environment.yml` and `requirements.txt` pin Python/runtime libraries. Tested on Linux x86-64 CPU, Python 3.11.5 and PyTorch 2.2.2+cpu. No CUDA, external simulator process, dataset, account or pretrained external policy is needed. PyBullet runs in DIRECT mode in the same process. The Darwin torch branch is provided but not independently tested. Internet is needed only for initial package installation. Verification records are in `verification/`.

The framework has **36 tests**; `tests/test_mission_progress.py` specifically checks visit-history visibility, the outstanding goal after reference advancement, and independence of Actor actions from Critic output. See `verification/` for the repository synchronization check, and rerun the commands above in your own environment.

## 2. Quick framework smoke run - random initialization

```bash
python run_learning_agent.py --total-timesteps 4096 --eval-every 1 --eval-episodes 1 --output runs/my_smoke
```

This executes actual simulation, rollout, GAE, policy/value gradient updates, validation, checkpoint writing and plotting. It uses **no supplied weights**. A completed smoke run is not evidence of a competent controller. Every output directory must be new to prevent overwriting results.

## 3. Train the mission agent

All main settings are explicit in `configs/mission_ppo.json`. To train from random weights:

```bash
python run_learning_agent.py --output runs/my_random_training
```

To repeat the successful PPO-only continuation stage using the included group-trained foundation:

```bash
python run_learning_agent.py --warm-start-pure models/foundation_ppo.pt --output runs/my_ppo_continuation
```

Both execute the same PPO objective. The foundation is NOT an imitation checkpoint: its 15-input policy was trained with the group's earlier PPO pipeline for 981984 physics steps at its selected checkpoint. The new 23-input layer copies/rescales those 15 input weights and initializes the eight added feature weights to zero; Adam restarts. This preserves deterministic initial actions, not critic calibration under the revised objective. The successful mission checkpoint is at 245760 new decisions / 490722 new physics steps, in a run continued to 400000 decisions / 799170 physics steps. See `models/README.md`.

To reproduce the two-stage procedure without any supplied weights:

```bash
python train.py --total-timesteps 300000 --n-steps 2048 --seed 7 --eval-every 20 --eval-episodes 3 --output runs/my_foundation_300k
python train.py --resume runs/my_foundation_300k/final.pt --total-timesteps 1000000 --output runs/my_foundation_1m
python run_learning_agent.py --warm-start-pure runs/my_foundation_1m/best.pt --output runs/my_mission
```

This reproduces the procedure; convergence from arbitrary initializations is not guaranteed. The older `train.py` uses the inherited 15-input policy; the new `run_learning_agent.py` / `pure_rl.train` use the 23-input mission policy. Do not mix their checkpoint formats or evaluation entry points.

**Budget units:** the new trainer counts total **policy decisions across eight environments**. A held action lasts up to two 0.05 s physics steps (policy 10 Hz, physics/inner loop 20 Hz). Real physics counts are logged separately. The old `train.py` uses one policy decision per physics step. Training wall time depends on the CPU; no short runtime promise is made.

**Restart limits:** the old foundation trainer supports `--resume` (fresh simulator episode). The new vectorized curriculum trainer saves initial/best/latest/final/interrupted weights, both Adam states and metadata, but does **not** expose exact mid-run resume or restore eight live physics states. Interrupted new-stage weights can be evaluated; restart the stage into a new output directory for further experiments. Do not present interrupted weights as a full simulator snapshot.

## 4. Evaluate a saved mission agent

```bash
python -m pure_rl.evaluate --checkpoint models/mission_ppo.pt --episodes 3 --seed-start 90000 --animation --output evaluations/my_replay
python -m pure_rl.evaluate --checkpoint models/mission_ppo.pt --episodes 50 --seed-start 70000 --output evaluations/reproduce_published_test
```

Evaluation always creates the original environment: original start at rest, 2 m/s timed path, all 10 ordered waypoints, 0.5 m **3-D** arrival tolerance, 45 s deadline and full wind/payload randomization. It does not use curriculum resets, training cutoff or shaped training reward. Actions are deterministic. Existing seed-70000 test results are historical published evidence; future model selection needs a NEW held-out seed set, not repeated tuning on these results.

Outputs include `episodes.csv`, `summary.json`, first-episode raw trajectory/plots and optional actual-PPO GIF. The original PID demonstration can be run with `python -m demo.demo_pid`; it is not PPO evidence. The old `evaluate.py` evaluates 15-input foundation checkpoints, not the mission checkpoint.

## 5. Framework contracts and organization

- `environment/`: inherited Oct-30 Gymnasium/PyBullet physics, wind, decreasing payload, timed path, raw reward and strict ordered success. Its source is unchanged.
- `rl/network.py`: inherited 15-64-64 actor and critic, squashed Gaussian log probability. `rl/ppo.py` supplies GAE; `rl/agent.py`, `rl/rollout.py`, root `train.py` retain the earlier foundation-training workflow.
- `pure_rl/agent.py`: extends that architecture to actor 23-64-64-4 and critic 23-64-64-1; checkpoint/config/provenance. `task.py`: public mission observation and training-only curriculum/reward/action hold.
- `pure_rl/collector.py`: eight independent PyBullet clients, batched inference, independent GAE per environment, persistent episode state across rollout boundaries.
- `pure_rl/learning.py`: PPO clipped policy gradient, on-policy value regression, separate Adam optimizers, gradient checks/clipping, KL early stop and learning-rate decay. No expert-action loss.
- `pure_rl/train.py`, `evaluate.py`, `plots.py`: full training workflow, validation-only best selection, original-task evaluation and real reward/loss curves.
- `run_learning_agent.py`, `configs/mission_ppo.json`: compact configured main entry point; `tests/`: inherited and added behavioral/integration checks.
- `evidence/`: previously measured positive AND negative results; `models/`: two PPO-only checkpoints; `INHERITANCE.json`: source hashes versus Oct-30 submission.

Observation: original 15 values + reference-velocity error (3) + next-unvisited-waypoint error (3) + ordered progress (1) + remaining time (1) = **23**. Wind, liquid mass and PID actions are excluded. Public mission inputs resolve a missing-goal/progress problem, but hidden disturbances still make this a partially observed task.

Actions are normalized thrust deviation, roll command, pitch command and yaw-rate command in [-1,1]. They map to nominal hover thrust +/-60%, roll/pitch +/-25 degrees and yaw rate +/-90 degrees/s. A shared inherited inner PD attitude controller applies body torques; the PPO outer loop does not output individual rotor speeds. "Pure PPO" means **no expert supervision of the outer policy**, not no classical control anywhere in the simulator.

Training logs separate `training_return` from original `raw_return`. Full-task validation uses the original unscaled return. Finite 45 s mission timeout is terminal in the new trainer, with remaining time observed. Training-only partial resets and early miss termination vanish by the final quarter; skipped prefixes are not counted as earned waypoints. Read `FRAMEWORK_NOTES.md` for the exact reward and reasoning.

## 6. Results and limitations

The locked teacher-free two-stage seed-7 policy completed 50/50 original-task held-out episodes (RMSE 0.829 m, raw return -4.960). The second continuation seed completed 36/50; both share one foundation. A direct random-start variant completed 0/50. Tuned classical control completed 50/50 with lower RMSE 0.639 m. Thus the framework can learn mission completion, but there is **no demonstrated advantage over tuned PID**, no reliable from-scratch convergence claim, and no coverage/deposition/real-flight proof. Controls and learning budgets differ. Detailed seed-level evidence and unsuccessful trials are included; the report states these limits.

The previous separate local imitation-assisted experiment is not the submitted training route. No behavior-cloning/DAgger code, demonstrations or imitation-trained weights are included here. Earlier reports did already disclose direct PyBullet, thrust-deviation actions and a shared inner controller; they did not say that their PPO was trained from expert actions.
