# Project Agent Scope and Decision Gate

## Project

This repository supports a CSCE 478/878 machine-learning project that classifies
short texts as human-written or AI-generated. The project uses the Kaggle
dataset `datascikhan/ai-vs-human-text-classification-benchmark-50k` and is
organized around the rubric sections in `rubric_explanation/`.

The current workstream is data understanding and preprocessing. The definition
of done for the preprocessing work is a reproducible, documented processed CSV
that can be consumed by later modeling code. It is not enough for a notebook to
describe transformations: the final artifact must be written to `data/processed/`
and its schema and row counts must be verified.

## Decision gate

No feature will be permanently removed, imputed, encoded, scaled, or designated
as a final model input without the project owner's explicit review and approval.
Exploratory summaries, plots, candidate flags, and reversible code changes are
allowed. They must be labeled as exploratory or proposed rather than final.

Before the final preprocessing CSV is produced, the owner must approve:

1. the prediction unit and target (`Label`),
2. whether raw text is used, transformed into text features, or retained only
   for traceability,
3. the treatment of identifiers and metadata,
4. the treatment of fields that may encode dataset construction or leakage,
5. missing-value, duplicate, outlier, categorical-encoding, and scaling rules,
6. the split strategy and which transformations are fit only on training data,
7. the final output columns and output filename.

## Working rules

- Keep `data/raw/` immutable. Derived files belong in `data/processed/`.
- Do not use the target, post-outcome information, or a field whose availability
  would differ at prediction time to create predictors.
- Use a stratified, reproducible split for classification unless a documented
  alternative is approved.
- Fit learned preprocessing steps on the training partition only; apply the
  fitted steps to validation/test data without refitting.
- Check duplicate texts and near-duplicates before splitting so that the same
  content cannot cross partitions unnoticed.
- Preserve a data dictionary and a transformation log in the rubric markdown.
- Report counts before and after every row-level operation.
- Do not infer that high univariate association proves a feature is valid. Check
  provenance, availability, and whether it represents a benchmark artifact.
- Prefer reversible changes and small, reviewable commits/patches.

## Ping-pong workflow

1. Inventory the dataset, schema, missingness, duplicates, class balance, and
   provenance patterns.
2. Produce reviewable plots and a candidate feature-risk table.
3. Ask the owner to approve or reject candidate treatments; do not finalize them
   in the agent's absence.
4. Implement the approved preprocessing pipeline.
5. Validate the processed CSV with schema, row-count, target, missingness, and
   reproducibility checks.
6. Update the rubric markdowns and changelog with the final rationale.

## Owner-approved direction for checkpoint 3

- Model inputs must be derivable from the submitted text snippet alone.
- `Raw_Text` remains in the processed CSV for traceability, but is not itself a
  tabular model input unless a separately documented text-vectorization model
  is approved.
- `Detection_Score_Benchmark_Model_A` is evaluation-only and must not enter the
  feature matrix.
- Dataset-construction metadata is excluded from the input feature policy.
- `Formal_Tone_Score` is excluded from the initial input feature policy because
  it is easily changed by prompting/style instructions and shows an unexplained
  construction-sensitive class pattern. This exclusion and its rationale must
  remain documented.
- The next review focuses on duplicate/near-duplicate risk, feature redundancy,
  and exact mathematical definitions for the remaining text-derived candidates.
- Because the public dataset sources do not disclose the original feature
  generator, the project will independently derive a documented candidate
  feature set from `Raw_Text`. Undocumented benchmark columns will not be
  treated as authoritative model inputs.

The independently derived candidate artifact and reduced processed CSV have
now been generated and validated:

- `data/processed/text_derived_features_candidate.csv` contains the full
  independently derived candidate feature set.
- `data/processed/human_ai_text_processed.csv` contains the reduced seven-feature
  model matrix plus traceability and target columns.

The preprocessing artifact is complete for handoff. Near-duplicate grouping and
the group-aware train/test split must be completed before model evaluation; they
are modeling controls rather than additional feature transformations.

## Approved reduced-feature policy

The current reduced policy keeps one representative from each major signal
family: word count, average word length, average sentence length, Guiraud root
TTR, punctuation density, Flesch-Kincaid grade level, and independently derived
burstiness. It removes highly redundant raw counts, the duplicate lexical
diversity measure, the constant special-character count, and average syllables
per word because syllable information is already incorporated into the retained
readability score. This policy is written by `src/prepare_final_data.py`; model
cross-validation will still compare it against the full candidate policy.

## Model-development handoff

The planned comparison includes Logistic Regression, Linear SVM, Random Forest,
Gradient Boosting, and K-Nearest Neighbors. All models must use the same
group-aware folds, leakage-safe preprocessing, and rubric metrics: accuracy,
precision, recall, F1-score, ROC-AUC, confusion matrices, and train/validation
comparisons. The full model rationale is in
`rubric_explanation/model_development.md`.

## Original-workstream checkpoint

The original near-duplicate grouping, group-aware splits and LR/RF baseline
evaluations are recorded in the reports. The active M4 continuation is described
below and in HANDOFF.md; do not repeat the original work as a new next step.

## Owner-approved M4 second-dataset policy (`better-data`)

The original benchmark workflow/results must be preserved in the final report.
The owner approved the following for the M4 Kaggle version-1 workstream:

- One passage or individual peer review per observation; do not concatenate reviews.
- Exclude null/empty entries and consolidate exact duplicate texts while preserving
  all retained origins in a separate provenance table.
- Keep related source passages and generated variants together across train/test
  and cross-validation boundaries.
- Use all nonempty, exactly deduplicated texts, without length cuts/downsampling.
- Keep the same seven independently derived features and retain raw text only for
  traceability; identifiers and source/generator metadata are not predictors.
- Approximately 80/20 stratified group-aware train/test split and five group-aware
  CV folds; fit any scaler on training folds only.
- Output `data/processed/m4/m4_processed.csv`, `m4_provenance.csv`, and
  `m4_splits.csv`; use separate report/model paths.

The owner reviewed the full-corpus diagnostics and approved a 0.85 word-trigram
cosine threshold for M4 near-duplicate grouping (13,799 combined groups, largest
216 texts). No rows are deleted by this grouping. Any changed threshold requires
review rather than being selected from model performance.
Do not treat an audit sample as an evaluation partition. Original model binaries
are absent in this checkout; any rebuild from saved settings must be identified
as a reconstruction and verified against saved original test predictions.

M4 preprocessing, the 74,757/18,678 split, original-model transfer, and M4 LR
training/evaluation are complete. The owner explicitly requested stopping M4 RF
training and handing that experiment to the partner. Do not automatically restart
it during this handoff. A partner can start the documented fresh RF command when
ready; there is no resumable partial search. Update README and final_project_report.md
together when adding results. Proposed LR class weighting, training-only threshold
selection and broader feature/quality experiments remain future work. The current
test results have been examined; avoid selecting follow-up changes from them.
