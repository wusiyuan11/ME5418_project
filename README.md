# ME5418_project

ME5418 Group 39 - aerial-spraying trajectory tracking under wind and decreasing liquid payload.

## Latest: Learning Agent milestone

The current 3-D Gymnasium/PyBullet PPO framework is in
[`ME5418_Group39_Learning_Agent/`](ME5418_Group39_Learning_Agent/README.md).
It builds on the group's environment/network milestones and adds sustained
teacher-free PPO training, checkpointing, reward logs and original-task evaluation.

The main design advance is exposing **actual ordered-waypoint completion** and the
**next outstanding goal** to the policy: the original 15 inputs become 23 public
mission inputs. The action interface remains four-dimensional. The original
time-driven reference and strict arrival criteria are retained, with their limits
explicitly discussed.

- [Install, train and evaluate](ME5418_Group39_Learning_Agent/README.md)
- [Annotated code guide](ME5418_Group39_Learning_Agent/CODE_GUIDE.md)
- [Shared two-page milestone report](ME5418_Group39_Learning_Agent/report/ME5418_Group39_Learning_Agent_Report.pdf)
- [Recorded results and limitations](ME5418_Group39_Learning_Agent/results/RESULTS_SUMMARY.md)
- [Chinese milestone summary](ME5418_Group39_Learning_Agent/本次提交说明.md)

The selected two-stage PPO completed 50/50 recorded original-task test episodes,
but the second continuation scored 36/50 and random-start training remained poor.
Tuned traditional control also completed 50/50 with lower tracking RMSE. These are
fixed-route, unequal-budget/control-rate experiments, not evidence that PPO beats
a strong PID baseline. The experiment data are dated 6 November 2026; this GitHub
synchronization does not constitute a new training result.

## Earlier prototype (preserved)

[`ME5418_Aerial_Spraying_Project/`](ME5418_Aerial_Spraying_Project/README.md)
contains the original 2-D point-mass prototype. It uses a different simulator and
action/observation interface; it has not been overwritten or presented as the
same environment as the new 3-D learning agent.

此次同步保留旧版，只新增第三次 Learning Agent 内容。相同实验记录集中保存一份，
未重复上传课程 ZIP、本机复检缓存或临时运行目录。课程提交包与本仓库的完整下载 ZIP
不是同一份材料；本仓库还包含历史原型。
