# Learning Agent for Aerial Spraying

ME5418 Group 39 | Shared learning agent milestone report

We extend the previous network demonstration into a runnable PPO agent for the spraying quadrotor. The main advance is making completed-waypoint history and the next outstanding goal visible to the policy, rather than relying on the moving reference error to describe the task.

## 1 Code explanation and learning agent design

### Main improvement since the previous submission: completed-waypoint inputs

The Oct-30 environment already counted actual ordered arrivals, but that history was absent from the 15 policy inputs. If the vehicle missed B while the timed reference moved to C, a small reference error could hide the uncompleted obligation. We now expose c/N, the completed-waypoint fraction, and waypoint[c] minus vehicle position, the next outstanding-goal error. The counter advances only through the original physical-arrival test in full-start missions.

| Item | Previous Neural Network submission | This Learning Agent submission |

| --- | --- | --- |

| Completed-waypoint history | Recorded inside the environment only | Observed by Actor and Critic as c/N |

| Next outstanding goal | Not supplied to the policy | 3-D error to waypoint[c], retained after a miss |

The observation is now 23 = 15 + 3 + 3 + 1 + 1: inherited state/tracking values, reference-velocity error, outstanding-goal error, completed fraction and remaining time. Actor 23-64-64-4 and Critic 23-64-64-1 retain the two 64-unit Tanh hidden layers. Both receive these observations; the Critic prediction is not an Actor input. No true wind, liquid mass or expert action is revealed.

The tanh-squashed Gaussian Actor commands thrust deviation, roll, pitch and yaw rate: hover thrust +/-60%, tilt +/-25 degrees and yaw rate +/-90 degrees/s. The shared inner PD applies torques; it supplies no expert labels.

### Training, feedback and code organization

Eight independent clients collect 256 decisions each. Per-client GAE feeds clipped PPO [1], with separate actor/value Adam rates 3e-4/1e-3, eight epochs, batch 256, clip 0.2, gradient limit 0.5, KL early stop and learning-rate decay [2]. The outer policy holds each action for up to two 0.05 s physics steps; gamma 0.9975 and lambda 0.97 apply per decision. Physics and decision counts are logged separately; the finite 45 s deadline is terminal.

Training adds goal progress and actual arrival-event rewards to 0.1 times the raw reward: +10 per arrival, +30 success and -30 timeout, with small goal-distance/yaw penalties. Early weak-wind and partial-route practice are withdrawn by the final quarter. Declared curriculum prefixes earn no reset reward and are logged separately from new arrivals. Shaping is heuristic, not proven policy-invariant; all evaluation uses the original full task.

environment/ and rl/network.py retain the previous sources. In pure_rl/, task.py constructs mission inputs, agent.py extends the network, collector.py preserves episodes, and learning.py updates PPO. train.py/evaluate.py save models and assess complete missions. run_learning_agent.py loads configs/mission_ppo.json. Core comments explain progress, GAE, reward definitions and train/test separation.

After creating and activating the Conda environment in README.md, a short run from random weights is:

python run_learning_agent.py --total-timesteps 4096 \
  --eval-every 1 --eval-episodes 1 --output runs/my_smoke

The pinned CPU runtime passes 36 tests, including new checks that identical motion with different visit histories produces different mission inputs. CSV/plots separate shaped training_return, raw_return and deterministic evaluation. New-stage exact live-simulator resume is not yet supported.


<!-- Page 2 -->


## 2 Existing codes and libraries

PyTorch 2.2.2 supplies layers/autograd/Adam; NumPy 1.26.4 handles calculations; Gymnasium 1.2.1 supplies the API [3]; PyBullet 3.2.7 integrates physics. Matplotlib 3.9.4/Pillow 12.3.0 produce visuals; pytest 8.3.5 runs checks. We implement PPO/GAE, the wrapper and workflow on the previous group code. No Stable-Baselines3, gym-pybullet-drones, expert demonstrations or external policy is used. environment.yml pins Python 3.11.5 and libraries; Linux CPU Conda recreation is verified. No GPU or separate simulator process is required.

## 3 Reflections and lessons learned

### What improved, and what the comparison means

The previous submission verified one update, not a trained success rate. The 15D baselines below were trained later. New candidates were selected on validation seeds 60000-60009, then locked before paired tests on 70000-70049. Tests retain the original rest start, full wind, 2 m/s route, 45 s deadline and ten ordered 3-D arrivals within 0.5 m, without training resets/cutoffs.

| Controller | Full-task success | Ordered points /10 | 3-D RMSE (m) | Original return |

| --- | --- | --- | --- | --- |

| 15D PPO: 300k run | 0/50 (0%) | 1.00 | 1.738 | -79.185 |

| 15D PPO: ~1M run | 0/50 (0%) | 2.00 | 0.572 | -24.764 |

| 23D PPO: continuation 7 | 50/50 (100%) | 10.00 | 0.829 | -4.960 |

| 23D PPO: continuation 8 | 36/50 (72%) | 9.40 | 0.936 | -16.017 |

| 23D PPO: random start | 0/50 (0%) | 2.00 | 2.633 | -107.414 |

| Tuned traditional PD/PID | 50/50 (100%) | 10.00 | 0.639 | 2.498 |

Waypoints, RMSE and return are episode means; success gives counts/percentages. Ordered counts include the start, counted once without reward. Return is the original evaluation reward, not shaped training reward.

Completion improved from 0/50 to 50/50 for the selected combined PPO route; tracking precision did not improve over tuned PID. Main-run full-task validation also progresses from 2.0 to 3.0, 6.2 and 10.0 points at 0, 81920, 163840 and 245760 new decisions (five fixed seeds). The same information is shown with separate completion/error and training-reward plots in results/.

The selected model continues a PPO-only foundation (981984 physics steps), adding 245760 decisions / 490722 physics steps at selection. Both continuation seeds share that foundation; direct random training fails. Observation-only trials also failed, so the gain cannot be assigned solely to added inputs. Budgets/rates and episode lengths differ. In 4.5 m/s base wind, PPO scores 17/20 versus tuned PID 20/20; 50/50 has a finite-sample 95% Wilson interval of 92.9%-100%.

### Proposal changes, remaining limits and next steps

Direct PyBullet, the 12 x 8 m field and hover-deviation actions are inherited simplifications. This milestone extends the proposed 15 inputs to 23 and changes training feedback/distribution, not the evaluation criteria. The reference still advances by time: exposing missed goals does not remove its conflict with returning to them. Next we will ablate waypoint inputs/rewards/curriculum, test independent foundations, improve corner/altitude control and compare progress-based or preview references. Route/speed and disturbance generalization remain open. The box model and waypoint metrics do not establish spray coverage, real energy savings or flight safety.

AI use declaration Codex assisted implementation, experiments, testing and writing. We retain the measured failures and limitations as part of the submitted work.

References [1] Schulman et al. Proximal Policy Optimization Algorithms (2017). https://arxiv.org/abs/1707.06347. [2] OpenAI Spinning Up, PPO. https://spinningup.openai.com/en/latest/algorithms/ppo.html. [3] Farama Foundation, Gymnasium API. https://gymnasium.farama.org/api/env/. [4] Group 39, Revised Proposal and Oct-30 Neural Network milestone report.
