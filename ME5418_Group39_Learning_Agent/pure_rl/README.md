# Mission PPO modules

The compact third-milestone entry point is `python run_learning_agent.py --output runs/new_run`, with settings in `configs/mission_ppo.json`. Read the root `README.md` for installation, random-start smoke training, two-stage PPO-only training, evaluation and restart limits. `FRAMEWORK_NOTES.md` explains the exact observation, action, reward, curriculum and proposal deviations.

`python -m pure_rl.train --help` lists the lower-level options, but its historical defaults differ from the selected mission configuration. Use the root entry point for the submitted settings. `python -m pure_rl.evaluate --help` describes evaluation of 23-input checkpoints on the original full mission. The historical all-candidate benchmark and report-generation scripts require the separate full research archive and are not included here; their measured CSV/JSON evidence is preserved under `evidence/`.
