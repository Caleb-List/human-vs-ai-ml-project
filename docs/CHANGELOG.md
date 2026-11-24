# Changelog

## Unreleased

### Added

- Complete the deferred M4 random forest: a fresh 36-configuration search on the approved features and folds (test ROC-AUC 0.9078, balanced accuracy 0.7153, Human false-positive rate 0.5356). It is better than M4 LR on every metric, with paired group-bootstrap intervals that exclude zero. Add it to the summary, report (section 11.7b), README, HANDOFF and evaluation doc, with an overall model recommendation (random forest). A full M4 rerun on a second computer reproduced the split and M4 LR results exactly.
- Add `project_notebook.ipynb`, one notebook covering rubric steps 1–4 for both datasets (replaces `notebooks/01`–`05`), with an M4 part and the final recommendation.
- Begin the exploratory M4 second-dataset workstream: version-pinned public Kaggle downloader with immutable raw-file verification, provenance-aware audit, separate feature sample and plots, parser tests, and a documented owner-review/partner handoff. Preserve the original benchmark workflow and results.
- Implement owner-approved M4 exact deduplication, empty-entry exclusion, seven-feature extraction and provenance-preserving source groups. Validate the 93,435-row processed CSV and 177,111 retained origins; complete exact near-duplicate review and owner-approved 0.85 grouping, with a validated 74,757/18,678 split and five group-aware CV folds.
- Complete original LR/RF reconstruction checks, no-refit transfer to M4, and tuned M4 LR. Add comparison graphics, domain error summaries, and paired whole-group bootstrap intervals. M4 RF retraining was stopped at the owner's request and deferred without a partial result.
- Add a study narrative to README, detailed M4 methods/results/improvement proposals to the final report, a pinned M4 environment, and HANDOFF.md with the partner's exact starting commands.

- Add `src/train_random_forest.py`: tuned random forest (max_depth x min_samples_leaf x max_features, 36 configs) on the same features, folds, selection metric, and stress test as logistic regression, plus permutation feature importance.
- Add `src/compare_models.py`: side-by-side metrics for every trained model, test-error overlap, and an exact McNemar test.
- Add `notebooks/01`–`05`, one per rubric step, committed with executed outputs.
- Save per-row held-out test predictions for each model in `reports/metrics/`.
- Add `--model` to `src/predict_text.py`.
- Add `ipykernel` and `nbconvert` to `requirements.txt`.
- Add `src/build_splits.py`: exact all-pairs word-trigram similarity, data-derived near-duplicate groups (threshold 0.82), template-family groups (threshold 0.60), and a stratified group-aware 80/20 train/test split with five CV folds, written to `data/processed/human_ai_text_splits.csv`.
- Document the finding that the benchmark consists of 11 single-class generator templates (6 Human, 5 AI).
- Add `src/modeling.py` with shared data loading, folds, metrics, and plots for all models.
- Add `src/train_logistic_regression.py`: tuned (C x L1/L2) logistic-regression baseline with CV, held-out test, template-family hold-out, and benchmark-detector reference results.
- Add `src/predict_text.py` to classify an arbitrary text snippet with the trained model.
- Fill in `rubric_explanation/model_eval_and_comparison.md` and the logistic-regression section of `model_development.md`.
- Add `scikit-learn`, `scipy`, and `joblib` to `requirements.txt`; ignore `/models/`.
- Organize EDA findings under `observations/`, beginning with the class-associated `Formal_Tone_Score` pattern and possible explanations.
- Add exploratory scatter plots for tone, burstiness, coherence, readability, and text length, plus a plot showing the formal-tone score ceiling by class.
- Add an AI formal-tone score histogram to assess the apparent 40-to-60 distribution.

### Changed

- `download_m4.py` falls back to KaggleHub when Kaggle's direct download returns 404 for anonymous requests; all 28 files match the recorded version 1 checksums. `data_loader.py` no longer fails when `data/raw/` already has files. `run.py m4` now includes the M4 random forest. Scripts print output paths relative to the project.
- Prepare the repository for peer review. Add `run.py`, one entry point for every task (`status`, `original`, `m4`, `figures`, `predict`, `test`) that skips steps that are already built. Shorten the README to a summary, results, reading guide and quickstart; the full study narrative stays in `final_project_report.md`. Move `rubric_explanation/*`, `observations/formal_tone_score.md`, `HANDOFF.md`, `CHANGELOG.md` and `project_rubric.pdf` into `docs/` with an index, and update all links. Add a table of contents to the report and turn the section 8 to-do list into a status table. No code, data, feature, split or result changes.

- Move CV summaries, the template-family hold-out, and the detector reference into `src/modeling.py` so all models share them (logistic-regression results unchanged).
- Clarify that the formal-tone pattern is unexplained and could be leakage, a dataset artifact, or a legitimate signal; avoid treating the cause as established.
- Expand the README with the visualization outputs and the caveat around interpreting the dataset's precomputed tone score.
