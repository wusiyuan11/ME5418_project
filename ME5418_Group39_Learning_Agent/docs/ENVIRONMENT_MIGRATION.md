# Reusing the previous Conda environment

The first two milestones shared `me5418-aerial-spraying`; the current YAML
defaults to the different name `me5418-learning-agent`. This comparison uses
the supplied October Neural Network specification, not an arbitrary installed
environment. The final release preserves the latest group YAML/requirements.

## Choose one target

Create an isolated current environment:

```bash
conda env create --file environment.yml
conda activate me5418-learning-agent
```

Reuse the earlier assignment environment without renaming it:

```bash
conda env update --name me5418-aerial-spraying --file environment.yml --prune
conda activate me5418-aerial-spraying
```

Refresh an already existing current-named environment:

```bash
conda env update --name me5418-learning-agent --file environment.yml --prune
conda activate me5418-learning-agent
```

Do not activate the new name if only the old environment exists. Close running
training processes and save an export outside the release directory before
changing a shared environment. `--prune` can remove Conda-managed extras; pip
extras may remain. A fresh environment is safer for unrelated work. No actual
old-name environment was modified as part of writing these instructions.

## Full declared delta

| Package / setting | Neural Network submission | Current package |
|---|---|---|
| Environment name | me5418-aerial-spraying | me5418-learning-agent |
| Python | 3.11.6 | 3.11.5 |
| torch | 2.2.2 | 2.2.2+cpu on non-Darwin; 2.2.2 on Darwin |
| numpy | 1.24.3 | 1.26.4 |
| matplotlib | 3.7.2 | 3.9.4 |
| pillow | 13.0.0 | 12.3.0 |
| pytest | 7.4.0 | 8.3.5 |
| contourpy | 1.3.2 | 1.3.3 |
| MarkupSafe | 3.0.4 | 3.0.3 |
| pyparsing | 3.0.9 | 3.3.3 |
| python-dateutil | 3.0.0 | 4.0.0 |
| PyYAML | Not declared | 6.0.3 |
| ReportLab | Not declared | 4.4.4 |
| charset-normalizer | Not declared | 3.5.2 |
| setuptools | Not explicitly pinned | 65.5.0 |
| six | 1.17.0 explicitly declared | No longer explicitly pinned |

All other matching declared pins are unchanged. Python/pip/setuptools also
appear as Conda dependencies rather than only pip requirements. The CPU wheel
index is retained. These are repins, not all upgrades or algorithmic changes.

PyTorch supplies layers, autograd and Adam. PyYAML verifies the environment
specification. ReportLab and charset-normalizer support document generation;
they do not change policy learning. Gymnasium 1.2.1 and PyBullet 3.2.7 are
unchanged. No SB3, new simulator or imitation-learning library was introduced.

## Verify after installation/update

```bash
python verify_setup.py --strict-versions
python -m pip check
python -m pytest -q
python -m tools.verify_submission
```

Do not edit dependency pins merely to match an arbitrary local installation.
The execution target is Linux x86-64 CPU; the Darwin wheel branch does not
prove macOS was tested. See `../verification/README.md` for the actual release
execution method and resource limits, separated from historical checks.
