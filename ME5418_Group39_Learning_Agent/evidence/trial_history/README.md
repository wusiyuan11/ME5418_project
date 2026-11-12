# Canonical records for repeated evidence

The following records were byte-identical in the verified course package. They
are stored once here rather than committed at several paths:

| Historical record | Canonical file in this directory tree |
|---|---|
| Main `pure_turn_repeat2_warm_seed7/config.json` | `../selected_run/config.json` |
| Main `pure_turn_repeat2_warm_seed7/training_summary.json` | `../selected_run/training_summary.json` |
| Main `pure_turn_repeat2_warm_seed7/validation.csv` | `../selected_run/validation.csv` |
| Reproduction `pure_repeat2_seed7_reproduction/config.json` | `../selected_run/config.json` |
| Reproduction `pure_repeat2_seed7_reproduction/validation.csv` | `../selected_run/validation.csv` |

The reproduction's unique training summary remains in its own folder. All other
trial configuration/summary/validation records are retained. At the parent
level, `../validation_selection.json` is the common model-selection lock that
was duplicated as `selection_used.json` in each original assessment folder.

These aliases do not alter training inputs or measured results. The generic
training, single-checkpoint evaluation and results-summary commands do not
depend on the omitted duplicate paths.
