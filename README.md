#  Breast Cancer Detection & Diagnostic Intelligence

### An Explainable Machine Learning Framework for Breast Cancer Classification

<p align="center">
  <strong>End-to-end Machine Learning | Statistical Analysis | Predictive Modeling | Explainable AI</strong>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10-blue?logo=python" />
  <img src="https://img.shields.io/badge/Scikit--Learn-ML-orange?logo=scikitlearn" />
  <img src="https://img.shields.io/badge/Pandas-Data%20Analysis-150458?logo=pandas" />
  <img src="https://img.shields.io/badge/Streamlit-Interactive%20App-FF4B4B?logo=streamlit" />
  <img src="https://img.shields.io/badge/Status-In%20Development-yellow" />
  <img src="https://img.shields.io/badge/License-MIT-green" />
</p>

---

##  Project Overview

Breast Cancer Detection & Diagnostic Intelligence is an end-to-end machine learning project designed to investigate diagnostic measurements and develop predictive models for classifying breast tumors as **Benign (B)** or **Malignant (M)**.

The project combines statistical investigation, exploratory data analysis, feature engineering, machine learning experimentation, model evaluation and explainability into a structured analytical pipeline.

The primary objective is to build a reproducible and interpretable classification framework that demonstrates how machine learning can support research into breast cancer diagnostic patterns.

> **Important:** This is an educational and research project, not a clinically validated diagnostic system. Predictions must not replace professional medical assessment.

##  Project Objectives

- Investigate diagnostic feature distributions and statistical patterns.
- Identify relationships and multicollinearity among predictors.
- Examine outliers without automatically discarding potentially meaningful observations.
- Engineer and select relevant predictive features.
- Develop and compare multiple classification algorithms.
- Evaluate models using clinically relevant classification metrics.
- Interpret model predictions using explainable AI techniques.
- Develop an interactive application for exploring model predictions.

##  Dataset Overview

**Dataset:** Wisconsin Diagnostic Breast Cancer (WDBC)

| Dataset Attribute | Description |
|---|---|
| Total observations | 569 |
| Predictor variables | 30 |
| Target variable | Diagnosis |
| Target classes | Benign (B), Malignant (M) |
| Missing values | 0 |
| Duplicate observations | 0 |
| Duplicate identifiers | 0 |

### Target Distribution

| Diagnosis | Observations | Percentage |
|---|---:|---:|
| Benign | 357 | 62.74% |
| Malignant | 212 | 37.26% |

### Feature Categories

The 30 numerical predictors describe cell nuclei characteristics across three measurement groups:

- **Mean Features:** Average characteristics of cell nuclei.
- **Standard Error Features:** Variability in the measurements.
- **Worst Features:** Largest or most extreme measurements.

Features include radius, texture, perimeter, area, smoothness, compactness, concavity, concave points, symmetry and fractal dimension.

##  Exploratory Data Analysis

The initial investigation includes:

- Dataset integrity and structural validation.
- Descriptive statistical analysis.
- Target class distribution.
- Feature distribution analysis.
- Benign vs. malignant comparison.
- Spearman correlation analysis.
- Multicollinearity investigation.
- Outlier detection using the IQR method.

### Key Analytical Findings

- Malignant observations exhibit higher average radius, perimeter and area measurements.
- Concavity and concave points show substantial differences between diagnostic classes.
- Several geometric measurements demonstrate strong inter-feature correlations.
- Area-related standard error measurements exhibit comparatively high outlier frequency.

##  Project Architecture

```text
Breast-Cancer-Detection-ML/
│
├── app/                    # Interactive application
│
├── data/
│   ├── raw/                # Original dataset
│   └── processed/          # Validated and cleaned data
│
├── models/                 # Trained model artifacts
│
├── notebooks/
│   └── 01_data_exploration.ipynb
│
├── outputs/                # Generated visualizations
│
├── reports/                # Analytical reports
│
├── src/
│   └── validate_data.py
│
├── .gitignore
├── requirements.txt
└── README.md
```

##  Technology Stack

| Category | Technologies |
|---|---|
| Programming | Python |
| Data Manipulation | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Statistical Analysis | SciPy |
| Machine Learning | Scikit-learn |
| Model Persistence | Joblib |
| Interactive Development | Jupyter Notebook |
| Application | Streamlit |
| Development Environment | VS Code, Git, GitHub |

##  Machine Learning Roadmap

| Phase | Description | Status |
|---|---|---|
| 01 | Environment & project setup | Completed |
| 02 | Dataset validation | Completed |
| 03 | Exploratory data analysis | Completed |
| 04 | Statistical & correlation analysis | Completed |
| 05 | Outlier investigation | Completed |
| 06 | Feature engineering & selection | Planned |
| 07 | Model development & comparison | Planned |
| 08 | Model evaluation & optimization | Planned |
| 09 | Explainable AI (SHAP) | Planned |
| 10 | Interactive Streamlit application | Planned |
| 11 | Deployment & documentation | Planned |

##  Model Evaluation Strategy

Models will be evaluated using multiple performance metrics rather than accuracy alone.

- Accuracy
- Precision
- Recall / Sensitivity
- Specificity
- F1-Score
- ROC-AUC
- Confusion Matrix
- Cross-validation performance

Special attention will be given to false-negative predictions because incorrectly classifying a malignant observation as benign is an important risk to investigate.

##  Explainable AI

The project will incorporate model interpretability techniques to investigate:

- Global feature importance.
- Individual prediction explanations.
- SHAP feature contributions.
- Relationships between diagnostic measurements and model outputs.

## Installation & Setup

### 1. Clone the repository

```bash
git clone https://github.com/nidhi-yadav20799/Breast-Cancer-Detection-ML.git

cd Breast-Cancer-Detection-ML
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

### 3. Activate the environment

**Windows PowerShell**

```powershell
.\.venv\Scripts\Activate.ps1
```

### 4. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 5. Start Jupyter Notebook

```bash
jupyter notebook
```

##  Current Development Status

The project has successfully completed its initial data validation and exploratory investigation stages.

The validated dataset contains 569 observations and 30 predictor variables, with no missing values or duplicate observations.

The next development milestone focuses on feature engineering, model experimentation and rigorous classification evaluation.

##  Author

**Nidhi Yadav**

Data Science Graduate | Machine Learning & AI Enthusiast

- GitHub: [@nidhi-yadav20799](https://github.com/nidhi-yadav20799)

---

<p align="center">
  <strong>Built with Python, Statistical Thinking & Machine Learning.</strong>
</p>

<p align="center">
  ⭐ If you find this project interesting, consider starring the repository.
</p>
