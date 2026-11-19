# Included group-trained PPO checkpoints

| File | Purpose | SHA-256 |
|---|---|---|
| foundation_ppo.pt | Teacher-free original 15-input PPO, selected at 981984 physics steps | 7d34f02f51aad669c455835cf7c315c0b3ad353dd86bccac739a6726e4dfe883 |
| mission_ppo.pt | 23-input goal-aware PPO continuation, selected at 245760 decisions / 490722 new physics steps | d3d1cec463fbf179001ff290192e896f83f37e7e2187a7911e1933f3feeeb605 |

Both checkpoints come from the group's local PPO experiments; neither is an external pretrained or imitation model. The mission checkpoint's training lineage totals 1472706 physics steps at selection. The continuation ran for 400000 decisions / 799170 new physics steps, and the selected model is not the last model. Failed trials and configuration selection consumed additional computation, so these figures do not establish equal-budget sample efficiency.

The default training entry uses random weights. Pass `--warm-start-pure models/foundation_ppo.pt` to continue from the foundation; the new stage starts with a fresh optimizer. Use `python -m pure_rl.evaluate` for `mission_ppo.pt` and the legacy `evaluate.py` for `foundation_ppo.pt`.
