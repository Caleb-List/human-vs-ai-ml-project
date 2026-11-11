# Model Development Plan

## Shared experimental design

The models will use the same reduced text-derived feature matrix, the same
group-aware splits, the same preprocessing conventions, and the same random
seeds. A second ablation will compare the reduced matrix with the full
independently derived candidate matrix. The benchmark detector score and all
dataset-construction metadata remain excluded from model inputs.

Near-duplicate texts will be assigned to groups before cross-validation. A
stratified group-aware procedure will preserve Human/AI proportions as closely
as possible in each fold. Any scaler, imputer, or feature-selection operation
will be fit inside the training fold only.

## Planned models

### 1. Logistic regression

This is the interpretable linear baseline. Standardization and L2
regularization will be used inside the pipeline. Coefficients provide a useful
directional interpretation of the seven summary features. The main tuning
parameters are regularization strength `C` and, if needed, class weighting.

### 2. Linear support-vector machine

This is a second linear margin-based model with a different optimization
objective from logistic regression. It is appropriate for a small, continuous
feature matrix and can be compared fairly under the same scaling and grouped
folds. The main tuning parameter is `C`.

### 3. Random forest

This nonlinear ensemble can represent threshold effects and feature
interactions without requiring scaled inputs. It provides a useful contrast to
the two linear models and supports feature-importance analysis. Important
parameters include number of trees, maximum depth, minimum leaf size, and
maximum features.

### 4. Gradient boosting

This nonlinear additive ensemble can capture structured interactions. Important
parameters include number of estimators, learning rate, maximum depth, and
subsampling. It will be tuned using only training folds.

### 5. K-nearest neighbors

This distance-based model tests whether nearby text-statistic profiles have
similar labels. It requires scaling and is likely to be sensitive to redundant
features, making it a useful diagnostic for the reduction decision. Important
parameters include neighbor count, distance metric, and weighting.

## Comparison and recommendation rule

All models will be evaluated with the rubric metrics: accuracy, precision,
recall, F1-score, ROC-AUC, confusion matrix, and training-versus-validation
performance. The recommended model will not be selected by accuracy alone. We
will consider mean and variation across grouped folds, evidence of overfitting,
interpretability, computational cost, and performance on the untouched test
set.

If the reduced feature matrix performs similarly to the full candidate matrix,
the reduced matrix will be preferred because it is simpler and easier to
explain. If a more complex policy provides a consistent multi-metric gain, that
tradeoff will be documented.
