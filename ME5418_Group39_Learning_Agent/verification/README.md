# Verification records and their scope

## Current release: 19 November 2026

`code_preservation.json` records incoming-archive identifiers and every protected
Python/config/model hash. All 45 Python files match the latest group source;
no extra Python module was added. Preservation is not an originality score.

`release_20261119/checks.json` describes current execution: strict versions,
pip dependency check, 36 tests, 23-input CPU demo, 4096-decision /
8115-physical-step random smoke training, published-50 reproduction, the
previous additional 9/10 check and separate 10/10 on 92000-92009. Episode CSVs
stay separate. No result selected/retrained the supplied checkpoint. The demo
JSON is included, not its untrained PT weights. Smoke completion is not convergence.

Fresh Conda recreation was attempted but failed during pip installation for
local disk capacity. The incomplete temporary environment was removed.
Successful checks used the existing pinned Python 3.11.5 / Torch 2.2.2+cpu
Conda runtime plus an isolated overlay of declared ReportLab 4.4.4 and
charset-normalizer 3.5.2. Strict verification checks active pins and pip check
passed. This is not a newly successful fresh Conda rebuild, a tested in-place
migration of someone's old environment, or cross-platform verification.
No member's pre-existing environment was renamed or pruned.

The six-page PDF/diagram are rendered and visually checked. The final ZIP is
independently extracted and verified again before delivery; the outer delivery
audit records its SHA-256 without putting the archive inside itself.

## Historical records, not current proofs

- `repository_sync_checks.json`: November 12 GitHub-sync checks.
- `quality_review/REVIEW.md` and `PROVENANCE_AUDIT.md`: supplied historical
  formatting/reproduction and limited traceability review, not a new originality
  guarantee. Verify any mentioned course threshold against actual course rules.
- `quality_review/fresh_10_episodes.csv`: 91000-91009, 9/10; seed 91009 timed
  out at 7/10 ordered points.
- `quality_review/reproduction_50_episodes.csv`: already-published seeds
  70000-70049, not 50 new independent episodes.
- Original research/selection data remain in `../evidence/`, with duplicated
  aliases mapped to canonical records.

Host-specific old logs, local runtimes and redundant weights are excluded.
Historical checks never substitute for checking the actual current build.

## Commands

```bash
python verify_setup.py --strict-versions
python -m pip check
python -m pytest -q
python -m tools.verify_submission
```

The project checker is unchanged; its regenerated inventory lists actual
delivered files and real current or historical report paths. Unbounded
observation-space and Matplotlib/pyparsing warnings remain disclosed, not
silently relabelled test failures. Exact live-client resume is unsupported.
