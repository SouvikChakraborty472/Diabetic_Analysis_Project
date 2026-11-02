# ================================================================
# STEP 0: IMPORT REQUIRED LIBRARIES
# ================================================================

import pandas as pd
import zipfile
# Import train_test_split to separate training and testing data.
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
# Import StandardScaler to standardize numerical features.
from sklearn.preprocessing import StandardScaler, TargetEncoder
# Import ColumnTransformer to preprocess numerical and categorical features.
from sklearn.compose import ColumnTransformer
# Import Pipeline to combine preprocessing and model training.
from sklearn.pipeline import Pipeline
# Import LogisticRegression for binary diabetes classification.
from sklearn.linear_model import LogisticRegression
import numpy as np
# Import metrics to evaluate classification performance.
from sklearn.metrics import (
    accuracy_score,          # Measures the proportion of correct predictions.
    precision_score,         # Measures the correctness of positive predictions.
    recall_score,            # Measures the proportion of positive cases detected.
    f1_score,                # Balances precision and recall.
    roc_auc_score            # Measures the model's ability to rank positive cases above negative cases.
)

# ================================================================
# STEP 1: IMPORT DATASET AND SELECT FEATURES
# ================================================================

# Replace with your actual file names
zip_path = 'Raw_Diabetes_Dataset.zip'

def read_zip_optimized(zip_file_path):
    # Open the zip file in read mode
    with zipfile.ZipFile(zip_file_path, 'r') as z:
        
        # Get the name of the file inside the zip 
        # (Using [0] assumes there's one main dataset inside)
        target_file = z.namelist()[0]
        print(f"Reading {target_file} directly from archive...")
        
        # Open the specific file within the zip as a stream
        with z.open(target_file) as f:
            # Read directly into Pandas
            # Note: Change read_csv to read_json/read_excel if your data isn't a CSV
            df = pd.read_csv(f)
            
    return df

# Load the data
dataset = read_zip_optimized(zip_path)
print(dataset.head())

# Select the same predictor columns used in your original code.
predictors = dataset.iloc[:, 2:7].copy()

# Extract the diabetes column as the target variable.
target = dataset["diabetes"].copy()

# Remove records where the target value is missing.
valid_rows = target.notna()

# Keep only records with valid target values.
predictors = predictors.loc[valid_rows].copy()
target = target.loc[valid_rows].astype(int).copy()

# Check that the target contains only binary labels, 0 and 1.
if not set(target.unique()).issubset({0, 1}):
    raise ValueError("The diabetes target must contain only 0 and 1.")

# Remove duplicate feature columns if any are present.
predictors = predictors.loc[:, ~predictors.columns.duplicated()]

# Display the selected predictor columns.
print("Selected features:", predictors.columns.tolist())

# Display the total number of usable records.
print("Total records:", len(predictors))

# Display the distribution of diabetes classes.
print("\nDiabetes class distribution:")
print(target.value_counts())

#print(predictors.head())
#print(target.head())
print(predictors.dtypes)
print(target.dtypes)

# ================================================================
# STEP 2: IDENTIFY FEATURE TYPES
# ================================================================

# Identify numerical columns for standardization.
numeric_features = predictors.select_dtypes(include=["number"]).columns.tolist()

# Identify categorical columns for one-hot encoding.
categorical_features = predictors.select_dtypes(
    include=["object", "category", "bool"]
).columns.tolist()

# Display the numerical features detected in the dataset.
print("\nNumerical features:", numeric_features)

# Display the categorical features detected in the dataset.
print("Categorical features:", categorical_features)

# ================================================================
# STEP 3: SPLIT THE DATASET
# ================================================================

# Reserve 20% of the data for final testing.
# Use the remaining 80% for cross-validation and bootstrap.
predictors_train, predictors_test, target_train, target_test = train_test_split(
    predictors,         # Input predictor variables.
    target,             # Diabetes target labels.
    test_size=0.20,     # Allocate 20% of records to testing.
    random_state=42,    # Make the split reproducible.
    stratify=target      # Preserve class proportions in both sets.
)

# Display the number of training records.
print("\nTraining records:", len(predictors_train))

# Display the number of testing records.
print("Testing records:", len(predictors_test))

print(predictors_train.head())
print(target_train.head())

# ================================================================
# STEP 4: CREATE THE LOGISTIC REGRESSION PIPELINE
# ================================================================

# Standardize numerical features and encode categorical features.
preprocessor = ColumnTransformer(
    transformers=[
        # Scale numerical features using training-data statistics.
        ("numeric", StandardScaler(), numeric_features),

        # Encode categorical features using target (mean) encoding.
        ("categorical", TargetEncoder(smooth="auto", cv=5, random_state=42), categorical_features)
    ],
    # Ignore any columns not explicitly selected for preprocessing.
    remainder="drop"
)

# Combine preprocessing and logistic regression into one pipeline.
model = Pipeline(
    steps=[
        # Apply the specified preprocessing operations.
        ("preprocessing", preprocessor),

        # Create the logistic regression classifier.
        ("classifier", LogisticRegression(
            max_iter=2000,        # Allow sufficient iterations for convergence.
            random_state=42       # Set a reproducible random seed where applicable.
        ))
    ]
)

# ================================================================
# STEP 5: FIT AND EVALUATE THE LOGISTIC REGRESSION MODEL
# ================================================================
# Train on the full training set.
model.fit(predictors_train, target_train)

# Predict labels and probabilities on the independent test set.
test_predictions = model.predict(predictors_test)
test_probabilities = model.predict_proba(predictors_test)[:, 1]

# Compute test-set evaluation measures.
test_metrics = {
    "accuracy":  accuracy_score(target_test, test_predictions),
    "precision": precision_score(target_test, test_predictions, zero_division=0),
    "recall":    recall_score(target_test, test_predictions, zero_division=0),
    "f1":        f1_score(target_test, test_predictions, zero_division=0),
    "roc_auc":   roc_auc_score(target_test, test_probabilities)
}

print("\n========== FIRST MODEL: TEST SET EVALUATION ==========")
for name, value in test_metrics.items():
    print(f"Test {name:<10}: {value:.4f}")

# ================================================================
# STEP 5: PERFORM 5-FOLD CROSS-VALIDATION
# ================================================================

# Create five stratified folds to preserve the class distribution.
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Define the metrics used to evaluate the classifier.
scoring = {
    "accuracy": "accuracy",     # Overall classification accuracy.
    "precision": "precision",   # Correctness of positive predictions.
    "recall": "recall",         # Detection rate for diabetic cases.
    "f1": "f1",                 # Balance between precision and recall.
    "roc_auc": "roc_auc"        # Ability to distinguish positive and negative cases.
}

# Cross-validate the same pipeline on training data only.
cv_results = cross_validate(
    model, predictors_train, target_train,
    cv=cv, scoring=scoring,
    n_jobs=-1, return_train_score=False
)

# Build a fold-wise table of all metrics.
cv_table = pd.DataFrame({
    metric: cv_results[f"test_{metric}"] for metric in scoring
})
cv_table.index = [f"Fold {i}" for i in range(1, 6)]

print("\n========== 5-FOLD CROSS-VALIDATION: ALL FOLDS ==========")
print(cv_table.round(4).to_string())

# Mean and standard deviation across folds.
cv_mean = cv_table.mean()
cv_std = cv_table.std(ddof=1)

print("\n---------- Mean +/- Std across folds ----------")
for metric in scoring:
    print(f"{metric:<10}: {cv_mean[metric]:.4f} +/- {cv_std[metric]:.4f}")

# ================================================================
# STEP 7: MODEL SELECTION STATEMENT
# ================================================================
# Compare test-set performance with CV performance.
# Selection rule: if test ROC-AUC and recall lie within 2 CV standard
# deviations of the CV mean, the model generalizes consistently.
auc_gap = abs(test_metrics["roc_auc"] - cv_mean["roc_auc"])
recall_gap = abs(test_metrics["recall"] - cv_mean["recall"])

auc_stable = auc_gap <= 2 * cv_std["roc_auc"]
recall_stable = recall_gap <= 2 * cv_std["recall"]

print("\n========== MODEL SELECTION ==========")
if auc_stable and recall_stable:
    print(
        "Selected model: Logistic Regression pipeline (StandardScaler + "
        "TargetEncoder + LogisticRegression), trained on the full training set.\n"
        f"Why: test ROC-AUC ({test_metrics['roc_auc']:.4f}) and recall "
        f"({test_metrics['recall']:.4f}) are consistent with the 5-fold CV means "
        f"({cv_mean['roc_auc']:.4f} and {cv_mean['recall']:.4f}), and fold-to-fold "
        f"variation is low (ROC-AUC std {cv_std['roc_auc']:.4f}). "
        "The model is stable and generalizes well to unseen data."
    )
else:
    print(
        "Selected model: None yet - Logistic Regression pipeline NOT approved.\n"
        f"Why: test performance deviates from CV performance "
        f"(ROC-AUC gap {auc_gap:.4f}, recall gap {recall_gap:.4f}) beyond "
        "2 CV standard deviations, indicating unstable generalization. "
        "Review features, class imbalance, or try other models."
    )

print("\nValidate the selected model on independent data before deployment.")