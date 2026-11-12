# Included group-trained PPO checkpoints

| File | Purpose | SHA-256 |
|---|---|---|
| foundation_ppo.pt | Teacher-free original 15-input PPO, selected at 981984 physics steps | 7d34f02f51aad669c455835cf7c315c0b3ad353dd86bccac739a6726e4dfe883 |
| mission_ppo.pt | 23-input goal-aware PPO continuation, selected at 245760 decisions / 490722 new physics steps | d3d1cec463fbf179001ff290192e896f83f37e7e2187a7911e1933f3feeeb605 |

These are produced by this group's local PPO experiments, not externally pretrained or imitation models. The mission checkpoint's total lineage at selection is 1472706 physics steps. The complete continuation ran for 400000 decisions / 799170 new physics steps; the best model is not the last model. More failed/configuration-selection attempts consumed additional computation, so this is not an equal-budget sample-efficiency claim.

The default training entry uses random weights. Explicit `--warm-start-pure models/foundation_ppo.pt` enables the disclosed two-stage route. The new training stage uses a fresh optimizer. Use `python -m pure_rl.evaluate` for `mission_ppo.pt`; use the legacy `evaluate.py` only for `foundation_ppo.pt`.
