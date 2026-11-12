# Previously measured experiment evidence

These are actual pre-packaging research records, not new training performed while preparing this submission. The compact code package preserves the main trainer and two useful checkpoints; the separate full Pure_PPO research archive retains all candidate weights and raw trajectories. Historical JSON paths refer to that original archive and are provenance, not portable command arguments. Included checkpoint hashes are recorded in `models/README.md`.

For GitHub, byte-identical repeated evidence is stored once. `validation_selection.json` is the common pre-test model lock used by both held-out and strong-wind assessments. The main seed-7 trial's config/summary/validation and the repeated run's identical config/validation are in `selected_run/`; see `trial_history/README.md` for the complete alias map. Unique trial summaries and all held-out episode rows remain present. The duplicate per-assessment `selection_used.json` files are intentionally not repeated.

- `selected_run/`: original main config, episode/update/validation CSVs, summary and reward/loss curves. `training_return` and `raw_return` have different definitions and must not be mixed.
- `trial_history/`: nine original configuration histories, including negative trials and both continuation seeds. These are combined design explorations, not nine RL algorithms or a controlled causal ablation.
- `validation_selection.json`: method selected using seeds 60000-60009 before held-out tests; training-time selection used 60000-60004. Success, ordered waypoints, fewer safety failures, then lower RMSE decide selection.
- `heldout_50/`: all 400 episode rows for eight controllers on paired seeds 70000-70049. Main pure PPO 50/50, second continuation 36/50, direct new-task random initialization 0/50, tuned classical 50/50. Mean timed-reference RMSE main 0.829 m vs tuned classical 0.639 m; raw return main -4.960 vs classical 2.498. Original unscaled reward and unchanged full mission only.
- `strong_wind_20/`: 60 paired episode rows for three controllers on seeds 80000-80019, max base wind 4.5 m/s rather than training 3 m/s, retaining the 0.7 m/s gust term. Main PPO 17/20 vs tuned classical 20/20; not used for selection.
- `replay/`: actual selected-policy first held-out episode, not PID footage.
- `historical_research_verification.json`: original research audit, including 29-test record and bit-identical full continuation reproduction. The submission's own tests are rerun separately in `verification/`.

50/50 has Wilson 95% interval 92.9%-100%, not a guarantee. Success episodes last about 34 s, timeouts up to 45 s; per-episode RMSE horizons differ. The main model still has average per-episode maximum timed error 1.961 m and visible altitude/corner deviations. No tuned-PID advantage, new-route generalization, real-flight safety, physical-energy or spray-uniformity result is established.
