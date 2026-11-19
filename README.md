# ME5418_project

ME5418 Group 39 - aerial-spraying trajectory tracking under wind and decreasing liquid payload.

## Latest: final team handoff (19 November 2026)

The current 3-D Gymnasium/PyBullet PPO framework is in
[`ME5418_Group39_Learning_Agent/`](ME5418_Group39_Learning_Agent/README.md).
It builds on the group's environment/network milestones and adds sustained
teacher-free PPO training, checkpointing, reward logs and original-task evaluation.

The final handoff addresses the TA's diagram and environment-reuse feedback.
All 45 Python files, both models, main JSON and dependency specifications are
preserved byte-for-byte from the latest group source ZIP. This is a
documentation/verification release, not new policy tuning.

- [Chinese team handoff](ME5418_Group39_Learning_Agent/START_HERE_组员必读.md)
- [Current shared technical PDF](ME5418_Group39_Learning_Agent/docs/ME5418_Group39_Technical_Handoff.pdf)
- [Actor-Critic diagram](ME5418_Group39_Learning_Agent/docs/actor_critic_architecture.png)
- [Conda migration guide](ME5418_Group39_Learning_Agent/docs/ENVIRONMENT_MIGRATION.md)

The shared PDF is not a substitute for required individual final reports.

The main design advance is exposing **actual ordered-waypoint completion** and the
**next outstanding goal** to the policy: the original 15 inputs become 23 public
mission inputs. The action interface remains four-dimensional. The original
time-driven reference and strict arrival criteria are retained, with their limits
explicitly discussed.

- [Install, train and evaluate](ME5418_Group39_Learning_Agent/README.md)
- [Annotated code guide](ME5418_Group39_Learning_Agent/CODE_GUIDE.md)
- [Historical two-page milestone report](ME5418_Group39_Learning_Agent/report/ME5418_Group39_Learning_Agent_Report.pdf)
- [Recorded results and limitations](ME5418_Group39_Learning_Agent/results/RESULTS_SUMMARY.md)
- [Chinese milestone summary](ME5418_Group39_Learning_Agent/本次提交说明.md)

The selected two-stage PPO completed 50/50 recorded original-task test episodes,
but the second continuation scored 36/50 and random-start training remained poor.
Tuned traditional control also completed 50/50 with lower tracking RMSE. These are
fixed-route, unequal-budget/control-rate experiments, not evidence that PPO beats
a strong PID baseline. The experiment data are dated 6 November 2026; this GitHub
synchronization does not constitute a new training result.

The later 91000-91009 check scored 9/10; that failure is retained. Final-release
checks reproduce the earlier results and separately record 10/10 on 92000-92009,
without changing weights or model selection.

## Earlier prototype (preserved)

[`ME5418_Aerial_Spraying_Project/`](ME5418_Aerial_Spraying_Project/README.md)
contains the original 2-D point-mass prototype. It uses a different simulator and
action/observation interface; it has not been overwritten or presented as the
same environment as the new 3-D learning agent.

此次在原有 Learning Agent 目录更新最终组内交接材料，不新增第二份 Final 代码目录。
保留最早原型和历史报告；相同实验记录只存一份，不重复上传源代码 ZIP、缓存或
临时训练目录。组内交接 ZIP 与整个仓库下载 ZIP 不同，后者还包含历史二维原型。
