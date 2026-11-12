# Human vs. AI Text Classification

Traditional machine learning project to distinguish between AI-written and human-written text.

## Requirements

- Python 3.11 or newer
- Internet access for the Kaggle download

## Set up a virtual environment

Run these commands from the repository root.

### Windows PowerShell

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### macOS or Linux

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Download and prepare the dataset

The `data/` directory is intentionally ignored by Git. A fresh clone will not
contain the raw or derived CSVs, so each collaborator must download and build
them locally. With the virtual environment activated, run:

```bash
python src/prepare_data.py
```

This downloads [`datascikhan/ai-vs-human-text-classification-benchmark-50k`](https://www.kaggle.com/datasets/datascikhan/ai-vs-human-text-classification-benchmark-50k) into `data/raw/`, validates the required fields, and writes a conservative prepared copy to `data/processed/AI_vs_Human_Text_Classification_Benchmark_processed.csv`. The analysis was developed against the local Kaggle version 1 download. If Kaggle serves a newer version, compare the schema and row counts before treating results as identical. The original CSV is never edited. Preparation preserves all rows, raw text, and missing metadata values; it converts `Label` to an integer and adds a readable `Label_Name` column.

Expected local files after this step:

- `data/raw/AI_vs_Human_Text_Classification_Benchmark.csv`
- `data/processed/AI_vs_Human_Text_Classification_Benchmark_processed.csv`

Most public datasets can be downloaded without authentication. If Kaggle requests authentication or dataset consent, sign in to Kaggle and configure a token using the [KaggleHub authentication guide](https://github.com/Kaggle/kagglehub#authenticate). You can also run `python -c "import kagglehub; kagglehub.login()"` to enter a token interactively. Keep tokens out of the repository.

## Explore the data

After downloading and preparing the dataset, print a summary of the raw CSV, including its dimensions, first rows, data types, class balance, missing values, numeric statistics, duplicate checks, constant columns, and source-model counts:

```bash
python src/explore_data.py
```

Create visualizations under `reports/figures/`:

```bash
python src/visualize_data.py
```

These commands operate on the downloaded raw CSV and do not train models.

Create an independently reproducible, text-only candidate feature artifact and
its review plots:

```powershell
python src/build_candidate_features.py
python src/visualize_candidate_features.py
```

The candidate extractor in `src/text_features.py` does not trust the source
CSV's undocumented engineered feature formulas. It derives auditable features
directly from `Raw_Text`; the resulting CSV and figures are intermediate review
artifacts for comparison with the reduced policy. The full 50,000-row extraction
prints progress in 1,000-row chunks and may take several minutes on a typical
laptop. Extracting features from one new passage is fast.

Expected candidate output:

- `data/processed/text_derived_features_candidate.csv`
- `reports/figures/candidate_text_feature_correlation_matrix.png`
- `reports/figures/candidate_text_feature_distributions.png`

## Write the reduced preprocessing artifact

After the candidate artifact has been created, write the reduced preprocessing
artifact:

```powershell
python src/prepare_final_data.py
```

This writes `data/processed/human_ai_text_processed.csv`. It retains
`Text_ID`, `Raw_Text`, and `Label` for traceability and includes seven reduced,
independently derived model features.

The reduced CSV is the preprocessing handoff for modeling. It is not a model
prediction file. Near-duplicate grouping and the group-aware train/test split
are performed by `src/build_splits.py`, not by this CSV-writing script.

## Build near-duplicate groups and the train/test split

```bash
python src/build_splits.py
```

This takes about two minutes. It scores every pair of texts by word-trigram
overlap, groups near-duplicates (similarity >= 0.82) and template families
(similarity >= 0.60), and writes a reproducible, stratified, group-aware
80/20 train/test split plus five CV folds to
`data/processed/human_ai_text_splits.csv`. Diagnostics go to
`reports/metrics/split_summary.json` and
`reports/figures/near_duplicate_similarity_diagnostics.png`. The feature CSV
is not modified. See
`rubric_explanation/data_preprocessing_and_exploration.md` for how the
thresholds were derived and for the finding that the benchmark is built from 11
single-class writing templates.

## Train and evaluate models

```bash
python src/train_logistic_regression.py
```

This tunes `C` and the penalty type with group-aware CV, scores the held-out
test set once, runs a template-family hold-out stress test, and writes metrics
to `reports/metrics/`, figures to `reports/figures/logistic_regression_*.png`,
and the fitted pipeline to `models/logistic_regression.joblib` (ignored by Git).
Results are summarized in `rubric_explanation/model_eval_and_comparison.md`.
New models should reuse the shared loaders, folds, and metrics in
`src/modeling.py` so comparisons stay fair.

## Classify a new text snippet

After training, score any text:

```bash
python src/predict_text.py "Paste a passage of text here."
```

It prints the predicted class, the probability that the text is AI-generated,
and the seven feature values it used. `--file path.txt` and standard input also
work.

The original visualization script writes class-balance, word-count, detector-score,
and source-model plots, plus scatter plots for word count vs. lexical diversity,
sentence length vs. readability, formal tone vs. emotion intensity, formal tone
vs. burstiness, coherence vs. burstiness, and emotion intensity vs. burstiness.
Scatter plots use an equal sample of up to 5,000 rows per class to reduce
overplotting. It also writes a chart showing how often the formal-tone score
reaches 100 and an AI-only histogram comparing score-bin counts against an
equal-count reference. These are exploratory views; no model training is part
of this stage.

## Dataset notes and project layout

In this dataset, `Label` 0 means Human and `Label` 1 means AI. Labels are balanced at 25,000 examples per class. Some metadata strongly identifies the source group: `Source_Model` is `Human_Origin` for every human sample and names the generating model for AI samples. `Model_Temperature` is missing for human rows, and `Contains_Code_Snippet` is constant. Keep these patterns in mind when interpreting the exploratory plots.

### Important observation: formal tone score

- `src/data_loader.py` downloads the dataset and loads the raw CSV.
- `src/explore_data.py` prints an exploratory summary.
- `src/visualize_data.py` writes charts to `reports/figures/`.
- `src/preprocessing.py` validates required fields and adds a readable label name while preserving the raw data and structural missing values.
- `src/prepare_data.py` creates the processed CSV under `data/processed/`.
- `src/text_features.py` independently derives auditable text-only features.
- `src/build_candidate_features.py` writes the full independently derived feature artifact.
- `src/visualize_candidate_features.py` creates candidate-feature review plots.
- `src/prepare_final_data.py` writes and validates the reduced seven-feature CSV.
- `src/build_splits.py` builds near-duplicate groups, template families, and the group-aware train/test and CV split.
- `src/modeling.py` holds the shared data loading, folds, metrics, and plots used by every model.
- `src/train_logistic_regression.py` tunes and evaluates the logistic-regression baseline.
- `src/predict_text.py` classifies an arbitrary text snippet with a trained model.
- `data/raw/` holds the original Kaggle file; `data/processed/` is for derived data. Git ignores all of `data/`.
