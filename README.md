# Breast Cancer Detection & Diagnostic Intelligence

### An Explainable Machine Learning Framework for Breast Cancer Classification

**Statistical Analysis | Predictive Modeling | Feature Engineering | Explainable AI**

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10-3776AB?logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/Pandas-Data%20Analysis-150458?logo=pandas&logoColor=white" />
  <img src="https://img.shields.io/badge/Scikit--Learn-Machine%20Learning-F7931E?logo=scikitlearn&logoColor=white" />
  <img src="https://img.shields.io/badge/NumPy-Scientific%20Computing-013243?logo=numpy&logoColor=white" />
  <img src="https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter&logoColor=white" />
  <img src="https://img.shields.io/badge/Status-In%20Development-yellow" />
</p>

---

## 01. Project Overview

Breast Cancer Detection & Diagnostic Intelligence is a machine learning project focused on investigating diagnostic measurements and developing predictive models to distinguish between benign and malignant breast tumors.

The project follows a structured data science workflow, beginning with dataset validation and statistical investigation, followed by exploratory data analysis, feature engineering, predictive modeling and model interpretability.

Rather than relying exclusively on classification accuracy, the framework emphasizes data quality, statistical relationships, feature redundancy, model reliability and the interpretation of predictive outcomes.

The ultimate objective is to develop a reproducible and interpretable classification pipeline supported by rigorous analytical methodology.

> **Research Disclaimer:** This project is intended for educational and research purposes. It is not a clinically validated diagnostic system and must not be used as a substitute for professional medical evaluation.

## 02. Project Objectives

- Establish a reproducible data validation pipeline.
- Investigate diagnostic feature distributions and statistical characteristics.
- Quantify differences between benign and malignant observations.
- Identify multicollinearity and redundant predictors.
- Investigate statistical outliers without automatically removing potentially meaningful observations.
- Develop meaningful feature representations and selection strategies.
- Experiment with multiple machine learning classification algorithms.
- Evaluate predictive performance using appropriate classification metrics.
- Apply explainability techniques to understand model decisions.
- Develop an interactive interface for model inference.

## 03. Dataset Overview

**Dataset:** Wisconsin Diagnostic Breast Cancer (WDBC)

The dataset contains measurements computed from digitized images of fine needle aspirates (FNA) of breast masses.

| Dataset Attribute | Value |
|---|---:|
| Total observations | 569 |
| Predictor variables | 30 |
| Target variable | Diagnosis |
| Diagnostic classes | 2 |
| Benign observations | 357 |
| Malignant observations | 212 |
| Missing values | 0 |
| Duplicate observations | 0 |
| Duplicate identifiers | 0 |

### Diagnostic Class Distribution

| Diagnosis | Count | Percentage |
|---|---:|---:|
| Benign (B) | 357 | 62.74% |
| Malignant (M) | 212 | 37.26% |

### Feature Structure

The 30 numerical predictors are organized into three measurement categories:

| Category | Description |
|---|---|
| Mean | Average characteristics of cell nuclei |
| Standard Error | Measurement variability |
| Worst | Largest or most extreme measurements |

The measurements cover:

- Radius
- Texture
- Perimeter
- Area
- Smoothness
- Compactness
- Concavity
- Concave points
- Symmetry
- Fractal dimension

## 04. Data Integrity & Validation

Before statistical investigation, the dataset undergoes structural and integrity checks.

The validation process includes:

- Dataset dimensionality verification.
- Removal of completely empty columns.
- Missing-value assessment.
- Duplicate observation detection.
- Duplicate identifier detection.
- Target class verification.
- Data type inspection.
- Predictor and target structure validation.

**Validation Results**

| Integrity Check | Result |
|---|---|
| Observations | 569 |
| Predictors | 30 |
| Missing values | 0 |
| Duplicate rows | 0 |
| Duplicate identifiers | 0 |
| Dataset validation | PASSED |

The original dataset contained one completely empty column (`Unnamed: 32`), which was removed during cleaning.

The validated dataset was saved separately to preserve the distinction between raw and processed data.

## 05. Exploratory Data Analysis

Exploratory analysis investigates the statistical characteristics of diagnostic measurements and their relationships with the target variable.

### Analysis Performed

1. Dataset structure and descriptive statistics.
2. Diagnostic class distribution.
3. Feature distribution analysis.
4. Benign versus malignant comparison.
5. Spearman rank correlation analysis.
6. Multicollinearity investigation.
7. Inter-feature correlation ranking.
8. Outlier detection using the Interquartile Range (IQR) method.

### Key Analytical Findings

**Diagnostic Feature Separation**

Malignant observations exhibit higher average measurements across several geometric features.

| Feature | Benign Mean | Malignant Mean |
|---|---:|---:|
| Radius Mean | 12.1465 | 17.4628 |
| Texture Mean | 17.9148 | 21.6049 |
| Perimeter Mean | 78.0754 | 115.3654 |
| Area Mean | 462.7902 | 978.3764 |
| Concavity Mean | 0.0461 | 0.1608 |
| Concave Points Mean | 0.0257 | 0.0880 |

The differences in concavity and concave points are particularly notable, suggesting that these measurements may contain useful discriminatory information.

**Correlation Investigation**

Spearman correlation analysis identified substantial relationships between several diagnostic predictors.

Selected observations:

| Feature Pair | Spearman Correlation |
|---|---:|
| radius_mean – area_mean | 0.9996 |
| radius_worst – area_worst | 0.9989 |
| radius_mean – perimeter_mean | 0.9978 |
| perimeter_mean – area_mean | 0.9971 |
| radius_worst – perimeter_worst | 0.9935 |
| perimeter_worst – area_worst | 0.9924 |

These relationships indicate considerable redundancy among certain geometric measurements.

This will be considered during feature selection and model development.

**Outlier Investigation**

Potential outliers were identified using the IQR method.

The investigation revealed that `area_se` had the highest flagged observation count, followed by `radius_se` and `perimeter_se`.

Importantly, statistical outliers are not automatically treated as erroneous observations. Their influence will be assessed during subsequent preprocessing and model evaluation.

## 06. Project Architecture

```text
Breast-Cancer-Detection-ML/
│
├── app/
│   └──                    # Interactive application
│
├── data/
│   ├── raw/
│   │   └── wdbc.csv       # Original dataset
│   │
│   └── processed/
│       └── wdbc_cleaned.csv
│
├── models/
│                          # Trained model artifacts
│
├── notebooks/
│   └── 01_data_exploration.ipynb
│
├── outputs/
│                          # Generated visualizations
│
├── reports/
│                          # Analytical reports
│
├── src/
│   └── validate_data.py
│
├── .gitignore
├── requirements.txt
└── README.md
```

## 07. Technology Stack

| Area | Technologies |
|---|---|
| Programming | Python 3.10 |
| Data Manipulation | Pandas, NumPy |
| Statistical Analysis | SciPy |
| Data Visualization | Matplotlib, Seaborn |
| Machine Learning | Scikit-learn |
| Model Persistence | Joblib |
| Interactive Analysis | Jupyter Notebook |
| Application Development | Streamlit |
| Development Environment | VS Code |
| Version Control | Git, GitHub |

## 08. Machine Learning Development Roadmap

| Phase | Development Activity | Status |
|---|---|---|
| 01 | Environment and project setup | Completed |
| 02 | Dataset acquisition