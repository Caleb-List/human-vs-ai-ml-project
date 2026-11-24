# Model Development Plan

## Shared experimental design

The models will use the same reduced text-derived feature matrix, the same group-aware splits, the same preprocessing conventions, and the same random seeds. A second ablation will compare the reduced matrix with the full independently derived candidate matrix. The benchmark detector score and all dataset-construction metadata remain excluded from model inputs.

Near-duplicate texts will be assigned to groups before cross-validation. A stratified group-aware procedure will preserve Human/AI proportions as closely as possible in each fold. Any scaler, imputer, or feature-selection operation will be fit inside the training fold only.

## Planned models

### 1. Logistic regression — implemented

Status: trained and evaluated (`src/train_logistic_regression.py`). Results are in `model_eval_and_comparison.md`.

**Why this model.** It is the interpretable linear baseline. With seven continuous summary features, a linear decision boundary is the simplest reasonable hypothesis, and its standardized coefficients show directly which text properties push a prediction toward AI or Human. Any more complex model must beat it to justify its added complexity.

**Assumptions and characteristics.** The log-odds of AI is modeled as a linear function of the features, `log(p / (1 - p)) = b + w · z`, where `z` is the standardized feature vector. Observations are assumed independent (which is why near-duplicates are kept on one side of every split). It is sensitive to feature scale, so a `StandardScaler` is fit inside the pipeline on each training fold only. Correlated features make individual coefficients less stable; the L1 option lets the tuner drop redundant ones.

**Pipeline.** `StandardScaler -> LogisticRegression(solver="saga", max_iter=5000, random_state=478)`.

**Hyperparameters and tuning.**

| Hyperparameter | Values searched | Role |
| --- | --- | --- |
| `C` | 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100, 1000 | Inverse regularization strength; small `C` = stronger shrinkage |
| `l1_ratio` | 0 (L2 penalty), 1 (L1 penalty) | L2 shrinks all weights; L1 can set weights exactly to zero |

16 configurations × 5 near-duplicate-group-aware CV folds, selected by mean validation ROC-AUC (`GridSearchCV`, `refit="roc_auc"`). Class weighting was not tuned because the classes are exactly balanced in every fold. `class_weight=None` is used.

**Selected:** `C = 0.1`, L1 penalty. Validation ROC-AUC reaches its plateau (about 0.9998) from `C = 0.01` upward for both penalties, and train and validation curves overlap at every `C` (`reports/figures/logistic_regression_tuning_curve.png`). The L1 penalty set the `Text_Guiraud_Root_TTR` coefficient to exactly zero, so lexical diversity adds nothing once the other six features are known.

### 2. Linear support-vector machine

This is a second linear margin-based model with a different optimization objective from logistic regression. It is appropriate for a small, continuous feature matrix and can be compared fairly under the same scaling and grouped folds. The main tuning parameter is `C`.

### 3. Random forest — implemented

Status: trained and evaluated (`src/train_random_forest.py`, `notebooks/04_random_forest.ipynb`). It was built second, ahead of the linear SVM, because it is the most different from logistic regression. Results are in `model_eval_and_comparison.md`.

**Why this model.** It is a nonlinear contrast to the linear baseline. Trees can represent threshold effects and feature interactions that a linear boundary cannot. If the forest found much more signal, that would show the linear model was leaving information unused. It also gives feature importances that can be checked against the logistic-regression coefficients.

**Assumptions and characteristics.** No distributional or linearity assumptions. Each tree is trained on a bootstrap sample and considers a random subset of `max_features` features at each split. Averaging many decorrelated trees reduces the variance a single deep tree would have. Splits depend only on feature order, so no scaling is needed and nothing is fit outside the model. Rows are still assumed independent, which is why the same near-duplicate-aware folds are used.

**Pipeline.** `RandomForestClassifier(n_estimators=300, random_state=478)`.

**Hyperparameters and tuning.**

| Hyperparameter | Values searched | Role |
| --- | --- | --- |
| `max_depth` | 4, 8, 16, None | Tree depth; deeper trees fit finer patterns and can memorize |
| `min_samples_leaf` | 1, 10, 50 | Minimum texts per leaf; larger values smooth the boundary |
| `max_features` | 2, 4, 7 | Features tried per split; 2 = floor(sqrt(7)) is the usual default, 7 is plain bagging |

36 configurations × the same 5 near-duplicate-group-aware CV folds, selected by mean validation ROC-AUC, the same rule as logistic regression. `n_estimators` is fixed at 300 rather than tuned: more trees reduce variance without overfitting, so it is a compute choice. `class_weight=None` because the classes are balanced.

**Selected:** `max_depth = None`, `min_samples_leaf = 1`, `max_features = 2`. Depth 16 and unlimited depth score identically. `max_features = 2` beats 4 and 7 at every depth and leaf size, so decorrelating the trees helps, though only by small margins because ROC-AUC is near its ceiling (0.9998–1.0000) for most configurations. Shallow trees (`max_depth = 4`) and large leaves (`min_samples_leaf = 50`) are mildly underfit (`reports/figures/random_forest_tuning_curve.png`).

### 4. Gradient boosting

This nonlinear additive ensemble can capture structured interactions. Important parameters include number of estimators, learning rate, maximum depth, and subsampling. It will be tuned using only training folds.

### 5. K-nearest neighbors

This distance-based model tests whether nearby text-statistic profiles have similar labels. It requires scaling and is likely to be sensitive to redundant features, making it a useful diagnostic for the reduction decision. Important parameters include neighbor count, distance metric, and weighting.

## Comparison and recommendation rule

All models will be evaluated with the rubric metrics: accuracy, precision, recall, F1-score, ROC-AUC, confusion matrix, and training-versus-validation performance. The recommended model will not be selected by accuracy alone. We will consider mean and variation across grouped folds, evidence of overfitting, interpretability, computational cost, and performance on the untouched test set.

If the reduced feature matrix performs similarly to the full candidate matrix, the reduced matrix will be preferred because it is simpler and easier to explain. If a more complex policy provides a consistent multi-metric gain, that tradeoff will be documented.
