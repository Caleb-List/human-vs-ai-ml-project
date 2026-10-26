# Important Observation: Formal Tone Score

## Observed distribution

In the local KaggleHub download (dataset version 1), `Formal_Tone_Score` differs sharply by class:

| Statistic | AI samples | Human samples |
| --- | ---: | ---: |
| Minimum | 40.001 | 37.379 |
| 25th percentile | 45.003 | 77.448 |
| Median | 50.099 | 100.000 |
| 75th percentile | 55.051 | 100.000 |
| Maximum | 59.999 | 100.000 |
| Exactly 100 | 0 / 25,000 | 14,343 / 25,000 (57.4%) |

The vertical stripe at 100 in the scatter plots is the Human samples sharing that exact value. It is a concentration of observations, not a plotting artifact. A histogram of the AI scores is saved as [ai_formal_tone_score_histogram.png](../reports/figures/ai_formal_tone_score_histogram.png).

The AI quartiles are close to those expected from a uniform distribution from 40 to 60 (45, 50, 55), and the histogram is approximately flat across 20 equal-width bins. This is consistent with a range-based generation process such as `rng.uniform(40, 60)`, but neither the quartiles nor histogram prove random generation. A real scoring formula could also yield a similar distribution.

## Possible explanations

The most plausible explanation is how the score was created or capped, rather than an inherent difference between human and AI writing. For example, a computed score could be clipped with `min(calculated_score, 100)`, causing values above 100 to pile up at 100. A default value, a scoring-system ceiling, bounded or random generation, a faulty formula, or different normalization between classes could also explain the pattern. The distribution alone cannot distinguish these possibilities, and it does not rule out a genuine text-related difference.

The [Kaggle dataset page](https://www.kaggle.com/datasets/datascikhan/ai-vs-human-text-classification-benchmark-50k) and [associated Kaggle notebook](https://www.kaggle.com/code/datascikhan/ai-vs-human-classification) do not document a formula for this score in the material we could inspect.

## How to use this field

A strong association with the label does not by itself prove target leakage. Leakage would mean the score was calculated using the human/AI label or information unavailable when classifying new text. A dataset-construction artifact could instead come from different collection or scoring procedures. Or, if the same independent method was applied to both classes, the score could reflect a legitimate signal in this dataset. We have not established which explanation applies.

A score of 100 identifies a Human sample in this dataset, but it does not identify every Human sample; the score ranges overlap. For now, preserve the column, flag it as suspicious, and exclude it from any future default predictor set pending clarification of how it was calculated:

> This field contains a strong class-associated pattern of unknown origin. It may reflect target leakage or a dataset construction artifact, so we exclude it from the initial predictor set pending clarification of how it was calculated.

The raw score and text remain unchanged in `data/raw/`.
