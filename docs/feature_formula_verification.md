# Feature Formula Verification

## Purpose

This document records the first attempt to verify whether equations proposed
for text-analysis features reproduce the corresponding columns in the local
Kaggle CSV. A formula is not accepted as the dataset's original implementation
merely because it is standard. Exact reproducibility also requires matching
tokenization, sentence segmentation, syllable counting, rounding, scaling, and
edge-case behavior.

## Results

| Proposed feature | Formula tested | Local result | Current status |
| --- | --- | --- | --- |
| Burstiness | `B = (sigma_L - mu_L) / (sigma_L + mu_L)` for sentence-length sequence `L` | A simple punctuation-based sentence splitter produced positive association with the stored column (`r` approximately `0.83`) but a different scale and values | Formula plausible; original segmentation/scaling not verified |
| Guiraud lexical diversity | `R = V / sqrt(N)` | Does not match `Lexical_Diversity_Score`; the stored column instead matches `Unique_Words_Count / Word_Count` exactly in the local data | Reject Guiraud as the stored-column explanation |
| Type-token ratio | `TTR = V / N` | Matches `Lexical_Diversity_Score` exactly using the stored `Unique_Words_Count` and `Word_Count` columns | Reproducible if the stored token/type counts are themselves verified |
| Flesch-Kincaid Grade Level | `0.39 * (words/sentences) + 11.8 * (syllables/words) - 15.59` | Very high but non-perfect agreement with the stored score (`r` approximately `0.995`; mean absolute difference approximately `0.385`) using stored average sentence length and syllables per word | Standard formula confirmed; stored implementation details remain unresolved |
| Functional grammar / stopword saturation | A density based on function-word or stopword counts | The dataset contains stopword count/density fields, but the screenshot does not uniquely define the stopword list or grammatical tagging method | Not verified until list, tokenizer, and implementation are identified |

## Detailed observations

### Burstiness

The proposed equation is a recognized burstiness form for a sequence of
lengths. Its result depends on whether `L` contains sentence token lengths,
character lengths, or another unit. It also depends on population versus sample
standard deviation and how punctuation boundaries are handled. The local test
used a simple `.`, `!`, and `?` sentence rule and word counts per sentence. The
result correlated with the stored `Burstiness_Score`, but the stored values were
on a different scale. The equation alone therefore does not verify the original
column.

### Lexical diversity

Guiraud's root type-token ratio is conventionally `V / sqrt(N)`. The local
column called `Lexical_Diversity_Score` does not match that quantity. Instead,
using the dataset's own `Unique_Words_Count` and `Word_Count` values gives:

`Unique_Words_Count / Word_Count == Lexical_Diversity_Score`

for the local rows, up to floating-point representation. This is ordinary
type-token ratio. The project must not describe this column as Guiraud's index
unless new evidence contradicts the local comparison.

### Flesch-Kincaid Grade Level

The standard equation in the screenshot is correct. A source implementation
such as `textstat` documents the same equation, but library behavior still
depends on its tokenizer, sentence counter, syllable algorithm, language, and
rounding. The local stored feature is close but not identical when calculated
from the stored intermediate averages. This suggests a different implementation
detail, rounding before aggregation, or a dataset-generation inconsistency.

### Stopword and functional-grammar features

Stopword density can be exactly reproduced only after the project identifies
the stopword vocabulary, case-folding rule, tokenization, and treatment of
contractions. Functional grammar density requires a part-of-speech tagger or a
fixed function-word list. Neither is specified by the screenshot alone.

## Research record and sources

- The [textstat implementation](https://github.com/textstat/textstat/blob/main/textstat/textstat.py)
  documents the Flesch-Kincaid equation and exposes implementation details for
  word, sentence, and syllable calculations.
- The [Quanteda lexical-diversity documentation](https://search.r-project.org/CRAN/refmans/quanteda.textstats/html/textstat_lexdiv.html)
  distinguishes ordinary TTR from Guiraud's root TTR and gives their formulas.
- The [Nature lexical-network article](https://www.nature.com/articles/s41599-023-02151-6)
  also states Guiraud's index as types divided by the square root of tokens.

## Source-reproducibility finding

The public Kaggle dataset metadata identifies the resource as an NLP text
classification dataset and provides general modeling examples, but it does not
publish the equations, tokenization rules, syllable algorithm, or feature-
generation source code for the 30-column benchmark CSV. The linked associated
notebook was also not available in a machine-readable form that exposed the
feature-generation implementation during this review. Consequently, the
original generator for several engineered columns cannot currently be
confirmed from public source material.

This is a reproducibility limitation of the source dataset, not a reason to
invent a formula. The project will cite the dataset, preserve the local raw
values, report this limitation, and independently implement only features whose
algorithms are fully specified and testable on arbitrary text.

## Decision rule

An existing feature can enter the final model matrix only if its algorithm is
documented, can run on arbitrary input text, agrees with known examples or the
local column within a stated tolerance, and passes redundancy and deployment
validity review. Otherwise, it remains exploratory or is excluded.
