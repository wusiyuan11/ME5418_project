# Mission PPO modules

The compact third-milestone entry point is `python run_learning_agent.py --output runs/new_run`, with settings in `configs/mission_ppo.json`. Read the root `README.md` for installation, random-start smoke training, two-stage PPO-only training, evaluation and restart limits. `FRAMEWORK_NOTES.md` explains the exact observation, action, reward, curriculum and proposal deviations.

Low-level `python -m pure_rl.train --help` exposes individual configuration flags; its historical defaults are NOT the selected mission configuration. Use the root configured entry for the submission's intended settings. `python -m pure_rl.evaluate --help` evaluates 23-input checkpoints on the unmodified full mission. The historical all-candidate benchmark/report generation scripts require the complete research archive and are intentionally not included in this lean submission; its measured CSV/JSON evidence is preserved under `evidence/`.
