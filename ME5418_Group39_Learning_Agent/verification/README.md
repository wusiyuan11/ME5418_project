# Repository synchronization verification

The source was synchronized from the verified 6 November Learning Agent materials
on 12 November 2026. The repository version omits host-specific packaging logs,
old repeated smoke-run artifacts and the duplicate full submission ZIP. It keeps
the commented code, exact dependency pins, two-page report, model checkpoints,
unique experimental evidence and result figures.

Recheck the framework from this directory:

```bash
python verify_setup.py --strict-versions
python -m pip check
python -m pytest -q --disable-warnings
python -m tools.verify_submission
python run_learning_agent.py --total-timesteps 4096 --eval-every 1 --eval-episodes 1 --output runs/new_smoke
python -m pure_rl.evaluate --checkpoint models/mission_ppo.pt --episodes 3 --seed-start 90000 --output evaluations/new_check
```

`repository_sync_checks.json` records the executed synchronization checks. These
short runs check execution, not fresh model selection or long-run convergence.
Training code, configuration, model bytes and experimental numeric data are
unchanged from the verified course version. Documentation paths, canonical
duplicate references and the repository content manifest are updated for GitHub.

Only Linux x86-64 CPU is independently tested. Gymnasium's unbounded-observation
warnings and Matplotlib/pyparsing deprecation warnings are known; no failed test
is hidden. Live eight-client simulator-state resume is not implemented, as
disclosed in the report and main README.
