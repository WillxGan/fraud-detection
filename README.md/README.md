# Credit-Fraud-Detector

## Dataset

This project uses the **Credit Card Fraud Detection** dataset from Kaggle.

- **Source:** https://www.kaggle.com/code/janiobachmann/credit-fraud-dealing-with-imbalanced-datasets/notebook

- **Description:** The dataset contains European card transactions from September 2013, with features transformed using PCA for confidentiality. Fraudulent transactions represent approximately 0.17% of all observations.
- **License:** Kaggle dataset terms apply.
> Note: The raw dataset is not included in this repository due to file size constraints.  
> To reproduce results, download the dataset from Kaggle and place `creditcard.csv` in `data/raw/`.

## Overview
This project builds a  fraud detection workflow using the Kaggle credit card transactions dataset. The pipeline covers preprocessing, proper evaluation for extreme class imbalance, threshold tuning, regression  interpretation (coefficients + odds ratios), and model comparison (SMOTE + Random Forest).

## Skills Demonstrated
- Load, inspect, and validate an imbalanced dataset (fraud cases are rare)
- Build repeatable ML pipelines (scaling + model)
- Evaluate correctly with imbalanced metrics (PR-AUC as primary)
- Perform threshold tuning and show tradeoffs 
- Interpret logistic regression using coefficients and odds ratios
- Compare imbalance strategies (class weights vs SMOTE)
- Compare linear vs non-linear models (Logistic Regression vs Random Forest)

## How It Works
1. **Load & inspect** the dataset and confirm class imbalance.
2. **Split the data** using a stratified train/test split.
3. Train a **baseline Logistic Regression** model using a pipeline (StandardScaler → LogisticRegression).
4. Evaluate ranking quality using **PR-AUC** (primary) and **ROC-AUC** (secondary).
5. **Tune the decision threshold** to understand false positive vs false negative tradeoffs.
6. Extract **coefficients + odds ratios** to interpret feature effects.
7. Compare imbalance handling with **SMOTE** (training data only).
8. Train a **Random Forest** model and compare performance and tradeoffs.

## Results Summary (Test Set)
### Baseline: Logistic Regression (class_weight="balanced")
- PR-AUC: **~0.72**
- Behavior at threshold 0.5: **high recall**, but **many false positives** (good for screening / review queues)

### SMOTE + Logistic Regression
- Confusion matrix (threshold 0.5) was **very similar** to baseline
- SMOTE did **not** meaningfully improve results here (class weighting was sufficient)

### Random Forest (class_weight="balanced")
- PR-AUC: **~0.86**
- ROC-AUC: **~0.96**
- Confusion matrix (threshold 0.5): **very high precision (~96%)** with **moderate recall (~76%)**
- Best for **high-confidence fraud flagging**, but misses more fraud than the logistic model at a recall-focused setting


## Project Structure
```text
fraud_detection/
├── data/
│   └── raw/                # raw dataset (do not commit large files)
├── notebooks/              # analysis notebook(s)
├── src/                    # helper scripts (optional)
├── models/                 # saved models (optional)
├── requirements.txt
└── README.md
