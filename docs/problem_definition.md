# Problem Definition and Data Understanding

## Problem statement

This project studies whether a text sample can be classified as human-written or AI-generated using observable text and dataset metadata. The prediction unit is one row/text sample, and the target is `Label`: `0` represents Human and `1` represents AI. This is a supervised binary classification problem.

The practical question is not simply whether a classifier can separate these rows. We want a detector whose inputs can be calculated from an arbitrary text snippet without knowing its source model, prompt, generation temperature, or the output of another detector. We also want to determine which signals are genuine properties of the text, which are artifacts of how this benchmark was assembled, and how much performance changes when potentially invalid or unavailable fields are excluded. The questions for the analysis are:

1. What are the dataset's class balance, text-length characteristics, metadata patterns, missing values, duplicates, and possible data-quality problems?
2. Which candidate predictors are available at prediction time and have a defensible relationship to the text rather than the data-collection process?
3. How do at least two supervised models compare under the same leakage-aware split and evaluation procedure?
4. Which limitations prevent the benchmark result from being interpreted as a general detector of all human and AI writing?

## Dataset and source

The data source is the Kaggle dataset [`datascikhan/ai-vs-human-text-classification-benchmark-50k`](https://www.kaggle.com/datasets/datascikhan/ai-vs-human-text-classification-benchmark-50k), downloaded through KaggleHub by `src/data_loader.py`. The local CSV contains 50,000 rows and 30 columns. The local preparation code keeps the raw CSV unchanged and writes a derived copy under `data/processed/`.

The classes are balanced in the local copy: 25,000 Human and 25,000 AI rows. `Text_ID` is unique for all rows, and `Raw_Text` is unique for all rows in the current audit. The local data contains no missing values in the required fields `Text_ID`, `Raw_Text`, and `Label`. `Model_Temperature` is missing for all Human rows and present for AI rows, so its missingness itself is class-associated.

## Target and candidate inputs

The target is `Label`. The dataset includes raw text, text-length and lexical statistics, readability measures, tone/style measures, a benchmark detector score, and generation metadata. The full candidate schema and initial risk notes are documented in [`data_preprocessing_and_exploration.md`](data_preprocessing_and_exploration.md).

The approved input boundary is text-only. `Raw_Text` remains in the processed CSV for traceability but is not automatically placed in the tabular feature matrix. Source metadata and `Detection_Score_Benchmark_Model_A` are excluded from model inputs; the benchmark score may be used only as an evaluation reference on held-out data. `Formal_Tone_Score` is excluded from the initial input policy because it is style/prompt-sensitive and has unexplained construction-sensitive behavior. Exact definitions for remaining candidate features are recorded in [`feature_derivation_spec.md`](feature_derivation_spec.md). The final subset still awaits the redundancy and duplicate review.

## Why machine learning is appropriate

The task has labeled examples and a categorical outcome, making supervised classification appropriate. A model can learn combinations of measurable text characteristics and provide a reproducible baseline for comparison. However, the benchmark is not proof that a model will generalize to future writing: the data-generation process may be easier to learn than authorship. This is why feature provenance, train/test separation, and comparison of alternative feature policies are central to the project.

## Data-quality concerns and limitations

- `Source_Model` is `Human_Origin` for every Human row and one of five named generators for AI rows. It may make the task a source-group classifier.
- `Generation_Prompt_Type` is `Human_Writing` for Human rows and generated prompt categories for AI rows. It may be unavailable for a real deployment.
- `Model_Size` and `RAG_Context_Provided` also show construction-related values that require provenance review.
- `Model_Temperature` has structural missingness: it is absent for Human rows.
- `Contains_Code_Snippet` is constant in the local data and cannot distinguish rows.
- `Detection_Score_Benchmark_Model_A` may duplicate the behavior of an existing detector and may not be a fair input if the goal is to build an independent classifier.
- `Formal_Tone_Score` has a strong, unexplained class-associated pattern, including a large Human concentration at exactly 100. It is preserved for audit but is not assumed valid without clarification of how this is calculated.
- Correlated engineered variables may count similar aspects of the same text multiple times; this is a modeling and interpretation concern, not by itself a reason to delete them.
- The public dataset description does not document the implementation of every engineered feature. Some original formulas therefore cannot be independently confirmed, limiting exact reproducibility of the source benchmark.

## Scope boundary for this checkpoint

This document defines the task and records risks. The column-removal and preprocessing decisions were made afterwards, following review of the exploratory figures and the feature-policy table. They are recorded in `data_preprocessing_and_exploration.md` and the final report.
