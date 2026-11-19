# Code quality, reproduction and provenance review

## Scope

The source ZIP was unpacked into separate baseline and revised directories.
The revised copy was scanned with sloppylint 0.5.1 (repository commit
`e0e3ced4f11c0533e1d8aa4c48d63fa6811f97ec`). Black formatted nine mission-agent,
entry-point and demo files. The original simulator and inherited network were
not reformatted, preserving their historical hashes. All 45 Python source
files were parsed and their syntax trees compared with the baseline: no
differences were found. This does not guarantee equivalence on every platform.

## Sloppylint findings

The scanner reported 22 findings both before and after formatting: six
`hallucinated_method`, eight `hallucinated_attribute`, six `single_method_class`
and two `deep_nesting`. The first two categories refer to the project's own
`collect()` methods and `length` attributes; those are valid Python APIs, not
Ruby or JavaScript syntax. The collector classes preserve episode state, and
controller classes are used through a common policy interface; their reported
method count is not by itself a defect. The two nesting notices refer to the
trajectory-output and optional-animation branches in `rl/evaluation.py`. They
remain a readability consideration, not a demonstrated runtime failure. The
scanner's score and verdict are heuristic; they do not measure authorship,
plagiarism, correctness or originality. No interfaces were renamed simply to
reduce the score.

## Execution checks on Linux x86-64

- A fresh Conda prefix was created from `environment.yml` using the libmamba
  solver. Strict version verification and `pip check` passed.
- The full test suite reported 36 passed; 246 warnings were reported, mainly
  Matplotlib/pyparsing deprecations and unbounded Gymnasium observation bounds.
- The baseline and formatted versions ran the supplied mission checkpoint on
  seeds 91000–91009. Their episode CSV files were byte-identical: 9/10 missions
  succeeded, and seed 91009 timed out after reaching 7/10 ordered points.
  `fresh_10_episodes.csv` preserves the formatted-version results. This is a
  small additional check, not a replacement for the held-out benchmark.
- The CPU network demo ran with 128 transitions and reported finite gradients
  and parameter changes in both branches. A separate one-episode evaluation
  with `--animation` generated a replay from seed 90000.
- Both versions ran the same random-initialized 4096-decision smoke training.
  The episode and validation CSVs and all 14 final model tensors matched
  exactly. The update rows matched apart from measured elapsed time. A short
  smoke run checks the training path; it does not demonstrate convergence.
- The supplied checkpoint was re-evaluated on the published seeds 70000–70049:
  50/50 succeeded, with mean tracking RMSE 0.8287559717601779 m. Success,
  ordered points, step counts and termination reasons matched all 50 historical
  rows. Numeric differences were limited to floating-point rounding; maximum
  return difference was approximately 3e-9. The reproduction rows are in
  `reproduction_50_episodes.csv`. These are the already-published seeds, not a
  new independent benchmark.

## What can and cannot be concluded about originality

The available inheritance record names 21 Python paths from the group's
earlier Neural Network milestone. In this version 18 are byte-identical to the
recorded earlier hashes; three have changed. Across the package there are 45
Python files. Raw physical-line counts, AI-style lint scores and matching an
earlier group submission cannot establish who wrote a line or whether it was
adapted from an external source.

The records identify the group's previous milestone code, the PPO/GAE method
references and the third-party libraries in `CODE_PROVENANCE.md`. They also
disclose Codex assistance. Limited exact-phrase web searches did not find matches,
but search coverage is incomplete and this is not a plagiarism scan. A credible
original-code percentage would require complete authorship records, a clear
definition of the denominator and an appropriate comparison against outside
sources. The group should verify all copied or adapted code and disclosures
before claiming compliance with the course's 50% requirement.
