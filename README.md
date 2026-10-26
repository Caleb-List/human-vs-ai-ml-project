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

## Download the dataset

With the virtual environment activated, run:

```bash
python src/prepare_data.py
```

This downloads the latest version of [`datascikhan/ai-vs-human-text-classification-benchmark-50k`](https://www.kaggle.com/datasets/datascikhan/ai-vs-human-text-classification-benchmark-50k) into `data/raw/`, validates the key fields, and writes a prepared copy to `data/processed/AI_vs_Human_Text_Classification_Benchmark_processed.csv`. The original CSV is never edited. Preparation preserves all rows, raw text, and missing metadata values; it converts `Label` to an integer and adds a readable `Label_Name` column. Both data directories are ignored by Git.

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

The script writes class-balance, word-count, detector-score, and source-model plots, plus scatter plots for word count vs. lexical diversity, sentence length vs. readability, formal tone vs. emotion intensity, formal tone vs. burstiness, coherence vs. burstiness, and emotion intensity vs. burstiness. Scatter plots use an equal sample of up to 5,000 rows per class to reduce overplotting. It also writes a chart showing how often the formal-tone score reaches 100 and an AI-only histogram comparing score-bin counts against an equal-count reference. These are exploratory views; no model training is part of this stage.

## Dataset notes and project layout

In this dataset, `Label` 0 means Human and `Label` 1 means AI. Labels are balanced at 25,000 examples per class. Some metadata strongly identifies the source group: `Source_Model` is `Human_Origin` for every human sample and names the generating model for AI samples. `Model_Temperature` is missing for human rows, and `Contains_Code_Snippet` is constant. Keep these patterns in mind when interpreting the exploratory plots.

### Important observation: formal tone score

In the local version 1 download, `Formal_Tone_Score` ranges from about 40 to 60 for AI samples, while 57.4% of Human samples have the maximum score of 100. The score's generation method is not documented in the material we could inspect. Its distribution may reflect capping, different feature-generation procedures, or a genuine signal; we have not established which. Preserve it, but exclude it from any initial predictor set until its origin is clarified. See the [observations folder](observations/README.md) and `reports/figures/ai_formal_tone_score_histogram.png` for details.

- `src/data_loader.py` downloads the dataset and loads the raw CSV.
- `src/explore_data.py` prints an exploratory summary.
- `src/visualize_data.py` writes charts to `reports/figures/`.
- `src/preprocessing.py` validates required fields and adds a readable label name while preserving the raw data and structural missing values.
- `src/prepare_data.py` creates the processed CSV under `data/processed/`.
- `data/raw/` holds the original Kaggle file; `data/processed/` is for derived data. Git ignores all of `data/`.
