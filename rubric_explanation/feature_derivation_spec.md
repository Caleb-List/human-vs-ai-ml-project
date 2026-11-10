# Text-Derived Feature Specification

## Purpose and boundary

Every model input must be computable from an arbitrary submitted text snippet.
The following definitions are intended to make feature construction auditable
and reproducible. They define candidate tabular features, not a final approved
feature subset. `Raw_Text` is retained for traceability, while IDs, generation
metadata, benchmark-detector output, and `Formal_Tone_Score` are outside the
input boundary.

## Shared notation and deterministic preprocessing

Let the submitted text be a Unicode string `s`. Before counting:

1. Replace each Unicode whitespace run with one ASCII space and trim leading
   and trailing whitespace.
2. Define a word token as a maximal contiguous sequence matching the Unicode
   letter/number categories. In an implementation, the tokenizer must be fixed
   and versioned; punctuation is not part of a word token.
3. Let `W = (w_1, ..., w_n)` be the resulting word sequence. If `n = 0`, all
   ratios below are stored as missing and the row is handled by the approved
   empty-text policy.
4. Define a sentence as a nonempty text span ending in `.`, `!`, or `?`, with
   consecutive terminators treated as one boundary. This is a simple deterministic
   rule, not a claim that it is linguistically perfect.
5. Let `U` be the set of lowercase word tokens, so `|U|` is the number of unique
   words. Lowercasing is Unicode-aware.

These rules are the project specification for new features. Existing dataset
columns whose original generator formulas are undocumented must not be called
exactly reproducible merely because their names look familiar. The first local
verification results are recorded in
[`feature_formula_verification.md`](feature_formula_verification.md).

## Candidate features and formulas

For text `s`, word sequence `W`, sentence sequence `S`, and punctuation count
`P`, the candidate definitions are:

| Feature | Exact definition |
| --- | --- |
| `Word_Count` | `n = |W|` |
| `Character_Count_Total` | Number of Unicode code points in normalized `s`, including spaces and punctuation |
| `Sentence_Count` | `|S|` under the sentence-boundary rule above |
| `Unique_Words_Count` | `|U|`, where `U = {lower(w): w in W}` |
| `Type_Token_Ratio` | `|U| / n` |
| `Guiraud_Root_TTR` | `|U| / sqrt(n)` |
| `Average_Word_Length` | `(sum length(w_i)) / n`, where length is Unicode code-point count |
| `Average_Sentence_Length_Words` | `n / |S|` |
| `Punctuation_Count` | Count of Unicode characters in the approved punctuation category set |
| `Punctuation_Density` | `P / max(1, Character_Count_Total)` |
| `Special_Character_Count` | Count of characters outside letters, numbers, whitespace, and the approved punctuation set |
| `Stopword_Count` | `sum_i I(lower(w_i) in D)`, where `D` is a versioned English stopword list |
| `Stopword_Density` | `Stopword_Count / n` |
| `Capital_Letter_Count` | Count of Unicode letters whose uppercase form differs from its lowercase form and which equal their uppercase form |
| `Average_Syllables_Per_Word` | `(sum syllables(w_i)) / n`; the syllable algorithm and version must be fixed and recorded because syllables are not directly observable from spelling |

For an arbitrary text detector, the implementation should prefer the features
whose definitions do not require an opaque external scorer. If readability,
syllable, emotion, coherence, or burstiness features are used, their exact
algorithm, package version, and fallback behavior must be added here before
they enter the final CSV. The current dataset does not document the generator
formulas for those columns, so their existing values are not automatically
accepted as reproducible inputs.

## Explicitly excluded or evaluation-only fields

- `Formal_Tone_Score`: excluded initially because style can be intentionally
  changed by prompting and the local distribution has an unexplained benchmark
  artifact. This is a validity decision, not an assertion that formality has no
  relationship to authorship.
- `Detection_Score_Benchmark_Model_A`: evaluation-only. It may be compared with
  model predictions on held-out data, but it is not available as a permitted
  input feature.
- `Source_Model`, `Model_Size`, `Model_Temperature`,
  `Generation_Prompt_Type`, and `RAG_Context_Provided`: excluded because they
  describe benchmark construction rather than arbitrary input text.
- `Text_ID`: traceability only.
- `Contains_Code_Snippet`: constant in this dataset and excluded; if code
  detection is later desired, it must be computed from the submitted text with
  a documented parser rather than copied from this metadata field.

## Redundancy policy to review

Correlation is a diagnostic, not an automatic deletion rule. For a numeric pair
`x` and `y`, Pearson correlation is:

`r_xy = sum_i ((x_i - mean(x)) (y_i - mean(y))) /
        sqrt(sum_i (x_i - mean(x))^2 sum_i (y_i - mean(y))^2)`.

The matrix should be reviewed alongside the feature formulas. For example,
word count, character count, sentence count, punctuation count, and stopword
count can all increase simply because the text is longer. If one member of a
highly redundant group is removed, the rationale must state whether the goal is
interpretability, numerical stability, or reducing duplicate information. A
model-based comparison of approved feature policies should be used when the
choice is not obvious.
