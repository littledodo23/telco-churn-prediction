# telco-churn-prediction
Leakage-safe ML pipeline predicting telecom customer churn — Logistic Regression, SVM, Random Forest, XGBoost &amp; LightGBM compared with tuning, class-imbalance handling, and a business cost-based evaluation.
# Telco Customer Churn Prediction

A complete, leakage-safe machine learning pipeline for predicting customer churn on the [IBM Telco Customer Churn dataset](https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco_customer_churn.csv), built for Birzeit University's Machine Learning course (Assignment 1).

The project covers the full ML workflow — EDA, preprocessing, feature selection, baseline modeling, five classification algorithms, hyperparameter tuning, class-imbalance handling, cross-validation, ROC/confusion-matrix analysis, a business cost-benefit study, a deliberate data-leakage demonstration, and a final deployment recommendation.

## Highlights

- **5 models compared**: Logistic Regression, SVM, Random Forest, XGBoost, LightGBM
- **Leakage-safe pipeline**: all imputation, scaling, encoding, feature selection, and resampling (SMOTE) are fit only on training folds via `sklearn.pipeline.Pipeline` / `imblearn.pipeline.Pipeline`
- **Class imbalance handling**: class weighting vs. SMOTE, compared against an untreated baseline
- **Hyperparameter tuning**: `RandomizedSearchCV` (15 iterations, stratified 5-fold CV) per model
- **Business-driven evaluation**: a $500 (false negative) vs. $50 (false positive) cost model drives threshold selection instead of relying on accuracy alone
- **Data leakage demo**: a deliberately leaked SMOTE-before-CV pipeline is compared against the correct one to quantify how much leakage inflates reported scores
- **Final recommendation**: LightGBM, selected on test ROC-AUC, F1, cost-weighted recall, and CV-to-test generalization gap — not on accuracy

## Results Summary

Final test-set performance (best pipeline per model):

| Model               | Accuracy | Precision | Recall | F1     | ROC-AUC |
|---------------------|----------|-----------|--------|--------|---------|
| Logistic Regression | 0.7381   | 0.5043    | 0.7834 | 0.6136 | 0.8407  |
| SVM                 | 0.6941   | 0.4575    | 0.8209 | 0.5876 | 0.8212  |
| Random Forest       | 0.7353   | 0.5009    | 0.7807 | 0.6102 | 0.8395  |
| XGBoost             | 0.7700   | 0.5496    | 0.7406 | 0.6310 | 0.8416  |
| **LightGBM**         | 0.7700   | 0.5496    | 0.7406 | 0.6310 | **0.8427**  |

**Recommended deployment model: LightGBM** — highest test ROC-AUC, tied-best F1, and the smallest CV-to-test gap (0.0002) of all five models, indicating the most reliable generalization. See the full report for the deployment justification and trade-off discussion against Random Forest and SVM.

## Repository Structure

```
.
├── notebooks/
│   └── Telco_Churn_Assignment.ipynb   # full pipeline: EDA → preprocessing → modeling → evaluation
├── report/
│   └── Telco_Churn_Report.pdf          # technical report with figures, tables, and discussion
└── README.md
```

## Dataset

The dataset is loaded directly from IBM's public repository — no manual download needed:

```python
import pandas as pd
url = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco_customer_churn.csv"
df = pd.read_csv(url)
```

It contains 7,043 customer records and 21 columns spanning demographics, account information, subscribed services, and billing details, with the binary target `Churn`.

## Methodology

1. **EDA** — data quality issues (e.g. blank strings in `TotalCharges`), class imbalance (~73.5% / 26.5%), feature distributions, and correlations
2. **Preprocessing** — median/most-frequent imputation, standard scaling, one-hot encoding, all inside a `ColumnTransformer` fit only on the training set
3. **Train/test split** — stratified 80/20 split, fixed before any feature selection or resampling
4. **Feature selection** — Mutual Information, Random Forest importance, and L1-regularized Logistic Regression, compared but not used to reduce the feature set (40 encoded features is small enough to keep in full)
5. **Baseline** — Dummy Classifier and plain Logistic Regression
6. **Modeling** — default 5-model comparison via stratified 5-fold CV, then `RandomizedSearchCV` tuning
7. **Imbalance handling** — class weighting vs. SMOTE (inside `imblearn.pipeline.Pipeline`)
8. **Evaluation** — ROC curves, confusion matrices, and a $500/$50 cost-asymmetry analysis to pick an operating threshold
9. **Leakage demonstration** — SMOTE applied before CV vs. correctly inside CV, showing an inflated ROC-AUC of 0.93 vs. the true 0.82
10. **Final comparison & recommendation** — best pipeline per model evaluated once on the held-out test set

## Requirements

```
pandas
numpy
scikit-learn
imbalanced-learn
xgboost
lightgbm
matplotlib
seaborn
```

Install with:

```bash
pip install -r requirements.txt
```

## Running

Open `notebooks/Telco_Churn_Assignment.ipynb` in Jupyter and run all cells top to bottom — a fixed random seed (`random_state=42`) is used throughout for reproducibility.

## Author

Danah Abu Rayya — Electrical and Computer Engineering, Birzeit University
