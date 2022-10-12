# Sparse-Projection-Infinite-Selection-Ensemble (SPISE)

* Code for the manuscript "Sparse Projection Infinite Selection Ensemble for Imbalanced Classification" (In Submission)

* Required Python 3 packages:
    1. numpy==1.21.5
    2. scipy==1.7.3

* Optional Python 3 packages: 
    1. `sklearn` (https://github.com/scikit-learn/scikit-learn)
    2. `imblearn` (https://github.com/scikit-learn-contrib/imbalanced-learn)
    3. `joblib` (for dataset loading, `datasets = joblib.load('MCIDatasets.pkl')`)

* SPISE is compatible with most sklearn APIs but is not strictly tested.

* Import: `from SPISEClassifier import SPISEClassifier`

* Train: `fit(X, y)`, with target $y_i \in \{0, 1\}$ as the labels.

* Predict: `predict(X)` (hard prediction), `predict_proba(X)` (probalistic prediction).

* Parameters: 
    1. `base_estimators`: dict, `default={'DT': DecisionTreeClassifier()}`, candidate classifier set $\mathcal{C}$, should have predict_proba() function"
    2. `n_estimators`: int, `default=30`, "the number of EHMCs $n$ in the FEHC"
    3. `population`: int, `default=10`, "the population size $\theta_P$ of the MCGA"
    4. `iteration`: int, `default=5`, "the number of iteration rounds $\theta_I$ of the MCGA"
