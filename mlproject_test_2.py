# -*- coding: utf-8 -*-
"""Machine Learning Project Script - Attempt 2 (Addressing Class Imbalance)

This script builds upon mlproject_test.py by attempting to address
class imbalance using the 'class_weight' parameter in SVC and RandomForest.

Changes:
- Added class_weight='balanced' to SVC and RandomForestClassifier.
- Added zero_division=0 to classification_report to handle warnings.

Steps:
1. Load data from 'comp_occur.csv'.
2. Encode string labels to numerical labels.
3. Split data into training and testing sets.
4. Define classification models (KNN, SVC, RandomForest) with pipelines.
   - SVC and RandomForest now use class_weight='balanced'.
5. Evaluate models using:
    a) 5-fold cross-validation (with scaling within folds).
    b) Performance metrics on the independent test set.
"""

# %% Imports
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
import warnings

# Suppress UndefinedMetricWarning for cleaner output if zero_division handles it
warnings.filterwarnings("ignore", category=UserWarning, module='sklearn.metrics._classification')


# %% Configuration
DATA_FILE = "comp_occur.csv"
TEST_SIZE = 0.25
RANDOM_STATE = 0
CV_FOLDS = 5

# %% Load Data
print("Loading data...")
dataset = pd.read_csv(DATA_FILE)
X = dataset.iloc[:, 1:].values  # Features
y_str = dataset.iloc[:, 0].values   # Labels (protein location) - Original string labels
print(f"Dataset shape: {dataset.shape}")
print(f"Features shape: {X.shape}")
print(f"Original Labels shape: {y_str.shape}")
print(f"Unique original labels: {np.unique(y_str)}")

# Encode string labels to numerical labels
le = LabelEncoder()
y = le.fit_transform(y_str) # Encode labels
print(f"Encoded Labels shape: {y.shape}")
print(f"Unique encoded labels: {np.unique(y)}")
print(f"Label mapping: {dict(zip(le.classes_, le.transform(le.classes_)))}\n")

# %% Split Data
print("Splitting data into training and test sets...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)
print(f"Training set size: {X_train.shape[0]}")
print(f"Test set size: {X_test.shape[0]}\n")

# %% Define Models and Pipelines
print("Defining models and pipelines (with class weights for SVC/RF)...")

# K-Nearest Neighbors (No class_weight option)
knn = KNeighborsClassifier(n_neighbors=18, metric='minkowski', p=2.5)
pipe_knn = Pipeline([
    ('scaler', StandardScaler()),
    ('knn', knn)
])

# Support Vector Classifier (with class_weight)
svc = SVC(kernel='rbf', random_state=RANDOM_STATE, decision_function_shape='ovo',
          probability=True, class_weight='balanced') # <<< Added class_weight
pipe_svc = Pipeline([
    ('scaler', StandardScaler()),
    ('svc', svc)
])

# Random Forest Classifier (with class_weight)
rf = RandomForestClassifier(n_estimators=10000, criterion='entropy', random_state=RANDOM_STATE,
                            n_jobs=-1, class_weight='balanced') # <<< Added class_weight
pipe_rf = Pipeline([
    ('scaler', StandardScaler()),
    ('rf', rf)
])

models = {
    "KNN": pipe_knn,
    "SVC_balanced": pipe_svc,         # Renamed for clarity
    "RandomForest_balanced": pipe_rf # Renamed for clarity
}

# %% Train and Evaluate Models
print("Training and evaluating models...")

for name, model_pipeline in models.items():
    print(f"--- Evaluating {name} ---")

    # --- Cross-Validation ---
    print(f"Running {CV_FOLDS}-fold cross-validation...")
    cv_scores = cross_val_score(estimator=model_pipeline,
                                X=X_train,
                                y=y_train,
                                cv=CV_FOLDS,
                                scoring='accuracy',
                                n_jobs=-1)

    print(f"CV Accuracy ({CV_FOLDS}-fold): {cv_scores.mean()*100:.2f} % (+/- {cv_scores.std()*100:.2f} %)")

    # --- Test Set Evaluation ---
    print("Evaluating on the independent test set...")
    model_pipeline.fit(X_train, y_train)
    y_pred = model_pipeline.predict(X_test)

    # Calculate metrics
    test_accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    # Added zero_division=0 to handle potential division by zero without warning
    class_report = classification_report(y_test, y_pred, zero_division=0,
                                         target_names=le.classes_) # Add target names

    print(f"Test Set Accuracy: {test_accuracy*100:.2f} %")
    print("Confusion Matrix:")
    print(conf_matrix)
    print("Classification Report:")
    print(class_report)
    print("-------------------------\n")

print("Script finished.")
