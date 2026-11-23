# Partner handoff: start here

## Checkpoint

The original benchmark experiment is preserved. M4 preprocessing, source and
near-duplicate grouping, the train/test and CV split, original-model transfer,
and M4 Logistic Regression are complete. **M4 Random Forest retraining was
stopped at the owner's request and is your next model experiment.** Its partial
search was not saved; start a fresh search. Do not treat the original RF transfer
result as a retrained M4 RF result.

Read the [README study narrative](README.md) first for context, then the
[final report](final_project_report.md) for detailed decisions and figures.
[M4 procedure and data dictionary](rubric_explanation/m4_dataset_review.md)
contains implementation details. Both main documents should be updated when new
results are added.

## First action on this computer

The approved local data and split already exist. From the repository root:

```powershell
.\.venv\Scripts\python.exe src/validate_m4.py
.\.venv\Scripts\python.exe src/train_m4.py --model random_forest --jobs 4
```

This trains 36 configurations across five stored group-aware folds (180 fits),
with 300 trees per forest. Four workers limit CPU/memory contention. It can take
substantially longer than the original benchmark's three-minute RF run. There
is no need to repeat feature extraction, regenerate splits, or rerun the full
similarity audit on this checkout first. The training script validates its input
checksums and split invariants before fitting.

After RF succeeds:

```powershell
.\.venv\Scripts\python.exe src/summarize_m4_results.py --bootstrap 200
```

Then update the results and interpretation in **both README.md and
final_project_report.md**, replacing the deferred RF status with measured scores,
CV train/validation comparisons, Human false positives, and the new paired
comparison. Preserve all original and transfer results.

## If you are using a fresh clone or another computer

`data/` and `models/` are ignored by Git. Code and reports alone do not contain
the processed CSVs, trained models, per-row M4 predictions, or similarity cache.
Follow the README's M4 reproduction commands, using Python 3.12 and
`requirements-m4-lock.txt`. Download/rebuild original benchmark features too if
reconstructing original models for transfer. The Kaggle M4 download is pinned
to version 1 and worked without credentials in this checkout.

The first exhaustive similarity comparison took about 33 minutes plus vector
construction. Reusing a checksum-matched local cache is much quicker. If local
artifacts are transferred between collaborators, validate them against their
manifests before use and resolve source redistribution terms; do not add raw
datasets or credentials to Git.

## Approved policy: keep this baseline reproducible

- One passage or individual review per observation; 0=Human, 1=AI.
- Exclude null/empty entries, consolidate exact text duplicates, preserve all
  retained origins separately. Keep distinct case/whitespace variants but group them.
- All 93,435 nonempty exact-unique texts; no length cuts or downsampling.
- Same seven documented features as the original experiment. Metadata, labels,
  prompts, IDs and detector scores are not predictors.
- Group known related sources and word-trigram cosine matches **>=0.85**.
  The owner explicitly approved this threshold after reviewing examples and group sizes.
- Seed 478; 74,757 training rows and 18,678 test rows; five training CV folds.
  Do not move related groups between partitions.
- Fit LR scaling inside training folds only. RF requires no scaling.
- New outputs remain under `data/processed/m4/`, `reports/m4/`, and `models/m4/`.

The group-aware split has 13,799 combined groups, largest 216 texts. Test
composition is 3,230 Human and 15,448 AI (82.71% AI). Every evaluation fold
contains both classes; no known-source/selected near-duplicate group crosses
train/test or training CV boundaries. The seven feature columns and the split
are not waiting on any additional approval. Changes to features, exclusions,
thresholds or split policy require owner review under AGENTS.md.

## What the completed results mean

| Model/evaluation | Accuracy | ROC-AUC | Balanced accuracy | Human false-positive rate |
| --- | ---: | ---: | ---: | ---: |
| Original LR transferred to M4 | 29.08% | 0.6221 | 53.74% | 8.58% |
| Original RF transferred to M4 | 19.42% | 0.6119 | 50.04% | 3.16% |
| LR retrained on M4 | 84.36% | 0.7103 | 56.20% | 86.84% |

The original benchmark scores were approximately 99.5% accuracy. Transfer failure
shows those scores do not establish general authorship detection. M4 LR ranking
improves after retraining, but its default threshold favors AI heavily. Always
predicting AI already obtains 82.71% accuracy. Its 99.25% AI recall and 0.913
AI F1 therefore do not make it a satisfactory detector of Human text.

LR selected `C=1000`, L1 by mean training-CV ROC-AUC. CV train/validation
ROC-AUCs are approximately 0.7130/0.7129, versus 0.7103 test. This is limited
discrimination with little observed train/validation gap, not evidence that a
near-perfect detector has been obtained. The completed LR search took about
7.8 minutes with two workers.

Original model binaries were absent, so transfer uses explicitly reconstructed
models with saved original settings and training membership. Both reproduce all
10,000 original saved test classifications. Versions and probability differences
are recorded; do not claim the historical binaries were recovered.

## Regression improvement experiments after the current comparison

1. Test `class_weight="balanced"` against the unweighted LR using the **existing
   training folds**. Record balanced accuracy, Human false-positive rate, AI recall,
   precision and ROC-AUC, rather than optimizing accuracy or AI F1 alone.
2. Choose a decision threshold using **out-of-fold training probabilities** and
   an owner-agreed false-positive/recall objective. Threshold changes can alter
   the confusion matrix, but do not improve ROC-AUC of fixed scores.
3. Inspect the saved grid. `C=1000` is an edge optimum, but a flat validation
   curve may make a larger search unhelpful. Compare mean and fold variation.
4. Review `test_errors_by_domain.csv` and consider text-length slices. Use these
   as descriptive error analysis, not as a basis for silently deleting test rows.
5. Propose domain/generator holdouts and separately reviewed full-candidate,
   interaction-feature or text-vectorization comparisons if needed. Do not change
   the approved seven-feature baseline while claiming a direct reproduction.

These are future proposals, not implemented performance improvements. The test
set has already been examined. Document follow-up design, tune only within
training folds, and use new independent evaluation for strong confirmatory
claims. Do not repeatedly choose features, weights, or thresholds from test scores.

## Artifacts to inspect

| File/location | Why it matters |
| --- | --- |
| `reports/m4/preprocessing_summary.json` | Transformation counts, feature schema, checksums |
| `reports/m4/validation_summary.json` | Executed CSV/provenance/feature validation |
| `reports/m4/similarity_review.json` | Full-corpus similarity diagnostics |
| `reports/m4/split_summary.json` | Actual partition/group/class counts and split hashes |
| `reports/m4/original_reconstruction.json` | Reconstruction checks against original predictions |
| `reports/m4/transfer_metrics.json` | No-refit original-model transfer results |
| `reports/m4/logistic_regression_metrics.json` | M4 LR settings, train/CV/test metrics, coefficients, runtime and versions |
| `reports/m4/model_comparison.csv` | Comparable completed stages; no invented deferred-model result |
| `reports/m4/group_bootstrap_intervals.json` | Whole-group paired uncertainty; conditional on fixed models/split |
| `reports/m4/test_errors_by_domain.csv` | Descriptive domain-level performance |
| `data/processed/m4/m4_provenance.csv` | Every retained raw origin mapped to canonical text |
| `models/m4/logistic_regression.joblib` | Completed M4 LR pipeline (local, ignored by Git) |

The summary handles the completed LR and both transfer models while RF is
deferred. Once RF artifacts exist, rerun it to include RF and paired comparisons.
Group-bootstrap intervals do not include retraining or split-selection uncertainty.

## Known limitations and final checks

Some M4 generations contain copied prompt material, repeated number sequences,
and entity-swapped templates. Source/model metadata is inconsistent in places.
No additional quality-based exclusions were approved. Grouping prevents shared
wording from crossing the approved boundaries but does not clean or validate
all content. The zero normalized exact overlap with the original corpus does
not prove semantic independence. Generator-held-out evaluation needs additional
metadata reconciliation. The dataset snapshot does not cover current generators
comprehensively.

Focused tests cover parsing, label conflicts, provenance bridges, exact similarity,
split boundaries and training-only scaling. Run them after implementation changes:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests
git diff --check
```

Commit/push the reviewed checkpoint through the normal Git workflow when ready
to share it. The scripts do not commit, push, or merge automatically.
