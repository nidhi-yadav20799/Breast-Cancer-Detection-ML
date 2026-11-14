# ONCOLENS — 2–3 Minute Demo Walkthrough

Use this outline to record a short screen walkthrough for the CODE-A-NOVA ML Developer Internship task. This document is a recording script, not a video recording.

## Before recording

1. Open the public Streamlit application: https://oncolens-breast-diagnostic.streamlit.app/
2. Open the GitHub repository: https://github.com/nidhi-yadav20799/Breast-Cancer-Detection-ML
3. Have the model-development notebook available.
4. Do not enter real patient information. Use only synthetic or clearly identified demonstration values.

## Suggested recording (2–3 minutes)

### 0:00–0:20 — Introduction

“Hello, this is ONCOLENS, a breast-mass classification research and educational prototype developed using the Wisconsin Diagnostic Breast Cancer dataset. The project compares machine-learning classifiers and provides an interactive Streamlit inference interface.”

### 0:20–0:45 — Dataset and EDA

Show the repository and the EDA notebook. Briefly point out the 569 observations, 30 numerical predictors, class distribution (357 benign and 212 malignant), feature distributions and correlation heatmap.

### 0:45–1:15 — Model development

Show the model-development notebook. Explain that five classifiers were compared using stratified five-fold cross-validation and that Logistic Regression and RBF-SVM were tuned with GridSearchCV. Mention that malignant-class recall and false negatives are important metrics in this task.

### 1:15–2:10 — Live application

Open the deployed Streamlit application. Show:
- Overview and recorded model metrics.
- Single-sample analysis using demonstration values only.
- Batch laboratory CSV validation and prediction workflow.
- Explainability outputs and methodology/limitations.

Do not present a model probability as a patient's medical risk or a clinical diagnosis.

### 2:10–2:40 — Wrap-up

“The saved model is a StandardScaler plus Logistic Regression pipeline. The project includes notebooks, validation code, saved model artifacts, evaluation outputs, a PDF report and a deployed prototype. The current holdout metrics are exploratory because the holdout comparison influenced model selection; a new untouched evaluation and clinical validation would be required before any real-world medical use.”

## Recording and sharing

- Keep the video between 2 and 3 minutes.
- Capture the browser and notebook clearly; avoid showing personal email, phone number, tokens or private account information.
- Upload to an approved shareable location (for example, an unlisted video or Loom link).
- Test the link in a private/incognito window to confirm that reviewers can access it.
- Share the link with CODE-A-NOVA only if supplementary submissions are accepted. The Google Form confirmation says not to submit the form more than once; do not submit a duplicate form.
