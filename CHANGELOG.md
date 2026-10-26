# Changelog

## Unreleased

### Added

- Organize EDA findings under `observations/`, beginning with the class-associated `Formal_Tone_Score` pattern and possible explanations.
- Add exploratory scatter plots for tone, burstiness, coherence, readability, and text length, plus a plot showing the formal-tone score ceiling by class.
- Add an AI formal-tone score histogram to assess the apparent 40-to-60 distribution.

### Changed

- Clarify that the formal-tone pattern is unexplained and could be leakage, a dataset artifact, or a legitimate signal; avoid treating the cause as established.
- Expand the README with the visualization outputs and the caveat around interpreting the dataset's precomputed tone score.
