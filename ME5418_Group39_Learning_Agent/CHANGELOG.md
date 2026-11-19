# Submission change log

## Final team handoff - 19 November 2026

- Added the Actor-Critic diagram, six-page shared technical note, Chinese team
  handoff and explicit response to the TA feedback.
- Corrected Conda environment names and documented declared version changes
  and explicit update targets. The group's dependency files are unchanged.
- Rebuilt the actual content inventory, labelled the earlier report historical
  and retained the supplied video. Seven identical evidence aliases stay deduplicated.
- Preserved all 45 Python files, main JSON and both models from the latest group
  source. No new project Python module, formatting or algorithm change was introduced.
- Reran versions, dependencies, 36 tests, CPU demo, short random training and
  checkpoint regressions. Fresh Conda recreation failed for local disk space;
  actual execution uses the existing pinned runtime plus isolated exact-version
  document packages. This is not labelled a successful new environment rebuild.
- Kept published 50/50, the earlier 9/10 and new 10/10 seed check separate;
  none is described as universal success or improvement caused by this release.

This log summarizes the changes represented in the available project records.
The experiment files under `evidence/` and the historical verification records
under `verification/` retain their original results and timestamps.

## Neural Network milestone to Learning Agent

- Extended the policy observation from 15 to 23 values to expose reference-velocity error, the next outstanding waypoint, completion progress and remaining time.
- Added sustained PPO training, a training-only reward and curriculum wrapper, separate actor and critic updates, model checkpoints and full-mission evaluation.
- Included the 15-input PPO foundation and the selected 23-input continuation checkpoint. The selected route uses PPO in both stages; it does not use expert-action demonstrations.
- Preserved recorded failed trials, validation results, held-out comparisons and stronger-wind evaluation. The successful continuation seeds share a foundation.

## Final-package preparation

- Updated the CPU network demo to exercise the 23-input mission actor-critic and record gradients and changes in both branches.
- Added instructions for updating the existing Conda environment, a dependency table and a code-provenance summary.
- Added the group's recorded results, checkpoint hashes and guidance for running validation. The final report and video are supplied outside this source archive.
- Edited the README, code guide, technical notes and submission explanations for clarity. This documentation edit did not retrain the policy or alter the recorded experiment data.

The recorded Learning Agent test runs and their scope are described in
`verification/README.md`. Run the documented checks again after unpacking the
archive; a historical test record is not proof that a different machine has
the correct environment.

## Code readability and verification

- Scanned the Python project with sloppylint and reviewed its findings. The
  retained findings and their limits are documented in
  `verification/quality_review/REVIEW.md`.
- Formatted the current mission-agent modules, configured entry point and
  network demo with Black. Python syntax-tree comparisons found no semantic
  changes; inherited environment and network files were left byte-identical.
- Recreated the Conda environment, ran the tests and compared seeded training
  and evaluation output before and after formatting. No supplied model weight
  or historical experiment data was changed.

## Public-source traceability review

- Added the public PPO/GAE method references and a PyTorch Tanh-transform
  formula correspondence to `CODE_PROVENANCE.md`.
- Documented the identified references and their limitations in
  `verification/quality_review/PROVENANCE_AUDIT.md`.
