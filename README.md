# ONCOLENS — Breast Cancer Detection & Diagnostic Intelligence

An educational machine-learning application for classifying Wisconsin Diagnostic Breast Cancer (WDBC) samples as benign (B) or malignant (M), using 30 numerical nuclear measurements.

> **Research disclaimer:** ONCOLENS is an educational prototype, not a clinically validated diagnostic device. It must not be used to diagnose, rule out cancer, guide treatment, or replace professional medical evaluation.

## Project overview

The repository contains an end-to-end data science workflow:
- Dataset validation and cleaning
- Exploratory data analysis (EDA) and statistical investigation
- Model experimentation and cross-validation
- A saved scikit-learn Logistic Regression pipeline
- Feature-importance outputs
- A Streamlit interface for single-sample and batch predictions

## Dataset

**Wisconsin Diagnostic Breast Cancer (WDBC)**

| Attribute | Value |
|---|---:|
| Observations | 569 |
| Predictors | 30 |
| Target | diagnosis (B / M) |
| Benign | 357 |
| Malignant | 212 |

The 30 predictors describe mean, standard-error and worst measurements for radius, texture, perimeter, area, smoothness, compactness, concavity, concave points, symmetry and fractal dimension.

## Repository structure

```text
.
├── app/
│   └── streamlit_app.py
├── data/
│   ├── raw/
│   │   └── wdbc.csv
│   └── processed/
│       └── wdbc_cleaned.csv
├── models/
│   ├── breast_cancer_logistic_regression.joblib
│   └── model_metadata.json
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   └── 02_model_development.ipynb
├── outputs/
│   ├── eda/
│   └── model_evaluation/
├── src/
│   └── validate_data.py
├── requirements.txt
├── requirements-dev.txt
└── README.md
```

## Methodology

1. Validate the raw WDBC dataset and remove the fully empty source column.
2. Explore class balance, feature distributions, correlations and potential outliers.
3. Split data using a stratified train/test split.
4. Compare classification algorithms using stratified cross-validation and tune selected candidates.
5. Persist the selected preprocessing/model pipeline with Joblib.
6. Use the Streamlit app for individual and CSV-based inference.

The saved application pipeline applies StandardScaler followed by Logistic Regression. The app uses the saved pipeline directly and does not retrain it.

### Model evaluation — important limitation

The current notebook compares the tuned Logistic Regression and SVM candidates on the holdout test split and then selects Logistic Regression. Consequently, the holdout split has influenced model selection; its reported metrics should be treated as **exploratory estimates, not a fully untouched final generalisation estimate**. A rigorous final evaluation should select the model using training/validation cross-validation, lock the model and threshold, and evaluate once on a new untouched test set.

The current reported Logistic Regression results on the existing split are:
- Accuracy: 97.37%
- Precision (malignant): 97.56%
- Recall (malignant): 95.24%
- Specificity: 98.61%
- F1-score: 96.39%
- ROC-AUC: 0.9964

These values describe this dataset split only. They do not establish clinical performance or real-world diagnostic accuracy.

## Streamlit application

The application includes:
1. **Overview** — dataset and saved evaluation summary
2. **Single-sample analysis** — enter all 30 measurements
3. **Batch laboratory** — upload a CSV and download predictions
4. **Explainability** — inspect saved coefficients and feature-importance outputs
5. **Model & methodology** — model pipeline, decision threshold and metadata

### Batch CSV format

Provide all 30 WDBC feature columns using their exact names. Optional columns are `id` and `diagnosis`. Diagnosis labels, when supplied, must be B/M or 0/1. Feature values must be numeric, finite and non-negative. Use the downloadable template in the Batch laboratory page as the column-name reference.

## Installation

Python 3.10 or newer is recommended. The saved model was serialized with scikit-learn 1.7.2; use the pinned runtime requirements to minimise version incompatibility.

```bash
git clone https://github.com/nidhi-yadav20799/Breast-Cancer-Detection-ML.git
cd Breast-Cancer-Detection-ML

python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS / Linux:
# source .venv/bin/activate

python -m pip install --upgrade pip
pip install -r requirements.txt
```

If the processed dataset is absent, place the original WDBC CSV at `data/raw/wdbc.csv` and run:

```bash
python src/validate_data.py
```

Start the application:

```bash
streamlit run app/streamlit_app.py
```

For notebook and EDA work, install the additional development dependencies:

```bash
pip install -r requirements-dev.txt
```

## Technology stack

Python · Pandas · NumPy · Scikit-learn · Joblib · Matplotlib · Streamlit · Jupyter

## Limitations and responsible use

- The model is trained and evaluated on the WDBC dataset; performance may not generalise to other populations, devices, laboratories or clinical settings.
- The application accepts measurements, not raw medical images.
- Predictions are statistical model outputs, not diagnoses.
- The dataset split used during development has influenced candidate selection; an independent final evaluation remains necessary.
- Do not upload identifiable patient information to this educational prototype.

## Project status

Core dataset validation, EDA, model experimentation, saved model artefacts and the Streamlit prototype are implemented. Reproducible final evaluation, broader external validation, deployment hardening and clinical validation are not complete.
