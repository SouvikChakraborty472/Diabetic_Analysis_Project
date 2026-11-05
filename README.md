# Diabetic Analysis Project

Welcome to the Diabetic Analysis Project repository! This project was created during my Data Science Internship at Exposys Data Labs. It showcases an end-to-end machine learning workflow for early diabetes prediction, from data loading and preprocessing to model building, cross-validation and model selection. Below you'll find detailed descriptions of the files included and an overview of the analysis conducted.

## Repository Contents

- **Model_Code.py**: The main model-building script for diabetes prediction, with a preprocessing pipeline, held-out test evaluation, 5-fold cross-validation and a model-selection statement.
- **pri_ana.py**: The initial, basic version of the model-building code. It has no model validation and is kept for reference and comparison.
- **Raw_Diabetes_Dataset.zip**: The compressed dataset (diabetes_prediction_dataset.csv) used for the analysis. The code reads it directly from the archive, so no manual extraction is needed.

## Project Overview

Diabetes is a chronic disease that is far easier to manage when it is detected early. To build a dependable prediction model, the project answers the following questions:

- Which patient indicators best describe a person's diabetes risk?
- How should numeric and categorical features be prepared for a logistic regression model?
- How well does the model perform on unseen data, beyond plain accuracy?
- Is the model stable across different subsets of the data?
- Which model should be selected, and why?
- What are the limitations before the model can be used in a real healthcare setting?

## Dataset

- **Size:** 100,000 patient records and 9 columns (gender, age, hypertension, heart_disease, smoking_history, bmi, HbA1c_level, blood_glucose_level, diabetes).
- **Target:** `diabetes` (0 = non-diabetic, 1 = diabetic).
- **Class distribution:** 91,500 non-diabetic (91.5%) and 8,500 diabetic (8.5%), so the target is imbalanced.
- **Features used by the model (5):**
  - Numeric: hypertension, heart_disease, bmi, HbA1c_level
  - Categorical: smoking_history
- **Not used in the current version:** gender, age and blood_glucose_level. These are candidates for the next iteration.

## Data Preparation and Modeling

This section focuses on preparing the data and building the model. The steps included:

- **Data Loading:** Reading the CSV directly from the zip archive.
- **Data Cleaning:** Removing records with a missing target, validating that the target is binary and removing duplicate feature columns.
- **Feature Typing:** Automatically separating numeric and categorical columns.
- **Data Splitting:** A stratified 80/20 train-test split (80,000 training and 20,000 testing records, random_state = 42).
- **Preprocessing Pipeline:**
  - StandardScaler for numeric features.
  - TargetEncoder (smooth = auto, cv = 5) for smoking_history, with internal cross-fitting to prevent target leakage.
- **Modeling:** Logistic Regression (max_iter = 2000), combined with the preprocessing into a single scikit-learn Pipeline so that all preprocessing is learned from training data only.

## Model Evaluation and Validation

Accuracy alone is misleading on imbalanced data, so the model is evaluated with several measures:

- **Metrics:** Accuracy, Precision, Recall, F1-score and ROC-AUC.
- **Held-out Test Set:** The baseline model is evaluated once on 20,000 unseen records.
- **Stratified 5-Fold Cross-Validation:** Run on the training data only, with metrics reported for every fold plus the mean and standard deviation.
- **Model Selection Rule:** The model is approved when the test ROC-AUC and recall lie within 2 cross-validation standard deviations of the CV means. A selection statement with the reasoning is printed automatically.

## Results

**Held-out test set**

| Accuracy | Precision | Recall | F1-score | ROC-AUC |
|----------|-----------|--------|----------|---------|
| 0.9434   | 0.8413    | 0.4118 | 0.5529   | 0.9066  |

**5-fold cross-validation (mean ± std)**

| Accuracy        | Precision       | Recall          | F1-score        | ROC-AUC         |
|-----------------|-----------------|-----------------|-----------------|-----------------|
| 0.9440 ± 0.0017 | 0.8512 ± 0.0149 | 0.4134 ± 0.0153 | 0.5564 ± 0.0162 | 0.9043 ± 0.0032 |

**Key observations**

- ROC-AUC above 0.90 shows strong separation between diabetic and non-diabetic patients.
- Test and cross-validation results agree closely, so the model is stable and does not over-fit.
- Precision is high (about 84%), meaning most flagged patients are truly diabetic.
- Recall is about 0.41, so the model currently detects roughly four in ten diabetic patients at the default 0.5 threshold.
- Accuracy is inflated by class imbalance, since a model predicting "non-diabetic" for everyone would already score 91.5%.

## Comparison with the Initial Version (pri_ana.py)

| Aspect | pri_ana.py (initial) | Model_Code.py (current) |
|--------|----------------------|-------------------------|
| Categorical encoding | LabelEncoder | TargetEncoder with smoothing and cross-fitting |
| Scaling | BMI only, fitted before the split | All numeric features, inside the pipeline |
| Data split | Random, no seed, no stratification | Stratified, fixed seed |
| Evaluation | Accuracy only | Accuracy, Precision, Recall, F1, ROC-AUC |
| Validation | None | 5-fold CV plus held-out test set |
| Model selection | Not applicable | Documented stability rule |

The initial version reported an accuracy of 0.94425 for Logistic Regression, 0.95485 for the Decision Tree and an RMSE of 0.2425 for Linear Regression. These values are not directly comparable and do not account for class imbalance, which is why the validated pipeline replaced them.

## Insights and Business Solutions

Based on the analysis, several key insights were identified, leading to actionable recommendations:

- **Early-Screening Support:** Use the model's risk probabilities to prioritise patients for laboratory testing and follow-up.
- **Threshold Tuning:** Lower the decision threshold or use class weights to raise recall, accepting more false positives in a screening setting.
- **Richer Predictors:** Add age and blood glucose level and compare against the current five-feature model.
- **Model Benchmarking:** Compare non-linear models, such as gradient boosting, against this baseline using the same validation protocol.
- **Probability Calibration:** Check calibration so predicted probabilities can be trusted as risk scores.
- **Clinical Validation:** Validate on independent patient data and review with clinicians before any deployment.

## Skills Demonstrated

- **Machine Learning:** Building and validating a binary classification model with scikit-learn.
- **Feature Engineering:** Target encoding, scaling and leak-free preprocessing with ColumnTransformer and Pipeline.
- **Model Evaluation:** Selecting appropriate metrics for imbalanced data, cross-validation and a held-out test set.
- **Data Manipulation:** Cleaning, validating and preparing healthcare data using pandas and NumPy.
- **Critical Thinking:** Identifying model limitations, such as low recall, and proposing practical next steps.
- **Reproducibility:** Fixed random seeds, stratified splits and a single pipeline object.

## Getting Started

**Requirements**

- Python 3.9 or later
- pandas, NumPy
- scikit-learn 1.3 or later (required for `TargetEncoder`)

**Installation**

```
pip install pandas numpy scikit-learn
```

**Run the project**

1. Clone or download this repository.
2. Keep `Raw_Diabetes_Dataset.zip` in the same folder as the scripts.
3. Run the main model:
   ```
   python Model_Code.py
   ```
4. (Optional) Run the initial version for comparison:
   ```
   python pri_ana.py
   ```
   Note that `pri_ana.py` reads the CSV from a local path, so update the file path in the script before running it.

**Note on warnings:** Recent scikit-learn versions show a deprecation warning for the `random_state` argument of `TargetEncoder`. It does not affect the results.

## Conclusion

This repository exemplifies my ability to take a healthcare problem from raw data to a validated machine learning model. The logistic regression pipeline discriminates well between diabetic and non-diabetic patients (ROC-AUC about 0.90) and performs consistently across cross-validation folds and an unseen test set. Because its recall is limited, it is selected as a validated baseline, and further tuning and independent validation are recommended before any clinical use. Feel free to explore the code and reach out if you have any questions or feedback.

## Contact

For any inquiries or further information, you can reach me at:

- Email: souvikchakraborty472@gmail.com
- LinkedIn: https://www.linkedin.com/in/souvikchakraborty472/

If you find it helpful, please give it a star and share it with others.

Thank you for visiting my repository! Happy analyzing!

## Acknowledgements

Special thanks to Exposys Data Labs for the internship opportunity and project guidance, and to the data science community for continuous support and inspiration.
