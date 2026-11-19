# Previously measured experiment evidence

Seven byte-identical aliases from the incoming ZIP remain stored once.
`trial_history/README.md` maps the five repeated main/reproduction records to
`selected_run/`; two historical `selection_used.json` copies map to
`validation_selection.json`. Unique data are retained without numeric edits.
November 19 regressions are separate in `../verification/release_20261119/`.

These research records predate packaging. This source archive includes the main trainer and two checkpoints; the separate full Pure_PPO research archive retains the candidate weights and raw trajectories. Paths in historical JSON files identify that original archive and are not portable command arguments. Checkpoint hashes are in `models/README.md`.

- `selected_run/`: original main config, episode/update/validation CSVs, summary and reward/loss curves. `training_return` and `raw_return` have different definitions and must not be mixed.
- `trial_history/`: nine original configuration histories, including negative trials and both continuation seeds. These are combined design explorations, not nine RL algorithms or a controlled causal ablation.
- `validation_selection.json`: method selected using seeds 60000-60009 before held-out tests; training-time selection used 60000-60004. Success, ordered waypoints, fewer safety failures, then lower RMSE decide selection.
- `heldout_50/`: all 400 episode rows for eight controllers on paired seeds 70000-70049. Main pure PPO 50/50, second continuation 36/50, direct new-task random initialization 0/50, tuned classical 50/50. Mean timed-reference RMSE main 0.829 m vs tuned classical 0.639 m; raw return main -4.960 vs classical 2.498. Original unscaled reward and unchanged full mission only.
- `strong_wind_20/`: 60 paired episode rows for three controllers on seeds 80000-80019, max base wind 4.5 m/s rather than training 3 m/s, retaining the 0.7 m/s gust term. Main PPO 17/20 vs tuned classical 20/20; not used for selection.
- `replay/`: actual selected-policy first held-out episode, not PID footage.
- `historical_research_verification.json`: original research audit, including 29-test record and bit-identical full continuation reproduction. The submission's own tests are rerun separately in `verification/`.

50/50 has Wilson 95% interval 92.9%-100%, not a guarantee. Success episodes last about 34 s, timeouts up to 45 s; per-episode RMSE horizons differ. The main model still has average per-episode maximum timed error 1.961 m and visible altitude/corner deviations. No tuned-PID advantage, new-route generalization, real-flight safety, physical-energy or spray-uniformity result is established.
