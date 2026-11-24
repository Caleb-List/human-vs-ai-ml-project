# Human vs. AI Text Classification

CSCE 478/878 project. Can simple, explainable measurements of a text snippet tell whether a human or an AI wrote it?

We compute **seven features** from the text alone (word count, average word length, average sentence length, Guiraud lexical diversity, punctuation density, Flesch-Kincaid grade, and burstiness), then train logistic regression (LR) and random forest (RF) classifiers. No metadata, detector scores, or labels are ever used as inputs.

## Results at a glance

| Experiment | Model | Accuracy | ROC-AUC | Human texts wrongly called AI |
| --- | --- | ---: | ---: | ---: |
| Original Kaggle benchmark (50k texts) | LR | 99.44% | 0.9998 | 0.68% |
| Original Kaggle benchmark (50k texts) | RF | 99.52% | 0.9999 | 0.74% |
| Original LR applied to M4, no retraining | LR | 29.08% | 0.6221 | 8.58% |
| Original RF applied to M4, no retraining | RF | 19.42% | 0.6119 | 3.16% |
| LR retrained on M4 (93k texts, 5 domains) | LR | 84.36% | 0.7103 | 86.84% |
| RF retrained on M4 (93k texts, 5 domains) | **RF** | **87.95%** | **0.9078** | **53.56%** |

**Takeaway.** The original benchmark is built from only 11 writing templates, so near-perfect scores there do not carry over to M4, a larger, independently collected multi-domain corpus. Retrained on M4, logistic regression underfits (ROC-AUC 0.71), while the random forest reaches ROC-AUC 0.91 and wrongly flags a third fewer Human texts. **We recommend the random forest** ([why](final_project_report.md#117b-m4-random-forest-result-and-model-recommendation)). Always guessing "AI" already scores 82.71% on M4, because it is 83% AI. Accuracy alone is therefore misleading, which is why the table also reports how often Human texts are wrongly called AI. Even the random forest still flags about half of Human texts at the default threshold.

![Model comparison](reports/m4/study_model_comparison.png)

## How to read this project

You can review everything on GitHub without installing or running anything.

1. **This README.** The summary and how to run the code.
2. **[Project notebook](project_notebook.ipynb).** One notebook covering rubric steps 1–4 for both datasets, saved with outputs: problem and data, preprocessing, both models, evaluation and comparison, then the M4 generalization test and the final recommendation.
3. **[Full report](final_project_report.md).** Every decision, figure, and limitation, including the M4 extension (section 11).
4. **[docs/](docs/README.md).** Detailed specs (feature formulas, data dictionary, M4 procedure), arranged by the question each one answers.

## Run it yourself

### 1. Set up (once)

Requires Python 3.12 and internet access for the Kaggle downloads. No Kaggle account is usually needed.

macOS / Linux:

```bash
python3.12 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
```

Windows PowerShell:

```powershell
py -3.12 -m venv .venv; .\.venv\Scripts\Activate.ps1; pip install -r requirements.txt
```

### 2. Use `run.py`

Every task goes through one script:

| Command | What it does | Time |
| --- | --- | --- |
| `python run.py status` | Shows what is already built and what to run next | instant |
| `python run.py original` | Downloads the 50k benchmark, builds features and the split, trains LR and RF, compares them | ~5–15 min |
| `python run.py predict "your text"` | Classifies a text snippet (needs `original` first) | instant |
| `python run.py m4` | Runs the full M4 extension: download, features, grouping, split, LR, transfer test, RF, summary | ~20 min–1 h |
| `python run.py figures` | Regenerates the exploratory figures | ~5 min |
| `python run.py test` | Runs the unit tests | seconds |

Steps whose output already exists are skipped, so a re-run after an interruption continues where it stopped. Add `--dry-run` to see the plan without running it, or `--force` to rebuild everything.

### 3. Classify your own text

```bash
python run.py predict "The mitochondria is the powerhouse of the cell, and it plays a key role in energy production."
```

Prints `Human` or `AI`, the probability of AI, and the seven feature values used. Other options: `--model random_forest`, `--file passage.txt`, or pipe text in on standard input. The model reflects this benchmark's idea of Human vs. AI writing and is **not** a general-purpose AI detector (see the results above).

### Troubleshooting

- **Kaggle asks you to sign in.** Run `python -c "import kagglehub; kagglehub.login()"`, or follow the [KaggleHub auth guide](https://github.com/Kaggle/kagglehub#authenticate). For M4, you can also download version 1 of the [dataset](https://www.kaggle.com/datasets/serjhenrique/m4-multi-generator-domain) in a browser and run `python src/download_m4.py --archive path/to/m4.zip`.
- **Re-running the notebook.** After `python run.py original`, run `jupyter nbconvert --to notebook --execute --inplace project_notebook.ipynb`. Part 5 reads the committed M4 results, so it doesn't need `run.py m4`.
- **Exact environment.** `requirements-m4-lock.txt` pins the exact package versions used for the M4 run (Windows, Python 3.12).

## Project layout

```text
run.py                  one entry point for every task (see table above)
project_notebook.ipynb  rendered walkthrough of rubric steps 1–4, both datasets
final_project_report.md full write-up
src/                    pipeline scripts; run.py calls them in order
  text_features.py        the seven feature formulas
  build_splits.py         near-duplicate grouping + group-aware split
  train_*.py              model tuning and evaluation
  *_m4.py                 M4 extension
reports/figures/        original-benchmark figures (committed)
reports/metrics/        original-benchmark metrics and test predictions (committed)
reports/m4/             M4 figures and metrics (committed)
docs/                   detailed specs, team handoff, changelog, course rubric
tests/                  unit tests for the M4 pipeline
data/, models/          created locally by run.py (not in Git)
```

## Method in brief

- **Text-only inputs.** All features are recomputed from raw text with documented formulas ([spec](docs/feature_derivation_spec.md)). The dataset's own engineered columns are not used because their formulas are undocumented.
- **Leakage controls.** Source model, temperature, prompt type, the benchmark's detector score, and `Formal_Tone_Score` are excluded ([why](docs/formal_tone_score.md)).
- **Near-duplicate-aware splits.** Similar texts (word-trigram cosine ≥ 0.82 for the original data, ≥ 0.85 for M4) are grouped, and a group never crosses train/test or CV folds. Scaling is fit inside training folds only.
- **Same protocol for every model.** Same folds, same tuning metric (ROC-AUC), and the test set is scored once.

## Limitations

Results describe these two datasets under this feature policy. They are not a claim of universal AI-text detection. Both M4 models use the default 0.5 threshold and no class weighting, so they over-predict AI: 87% (LR) and 54% (RF) of Human texts are wrongly flagged. Class weighting and threshold selection are proposed next steps ([handoff](docs/HANDOFF.md)), not completed results. M4's generators are also older (for example GPT-3.5 and davinci-003).
