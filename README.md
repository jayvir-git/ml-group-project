# 🔬 Protein Subcellular Localization Prediction 🧬

---

## 🌟 Project Overview

This project implements a machine learning pipeline to predict the subcellular localization of Gram-positive bacterial proteins. Given sequence-derived features (amino acid composition and occurrence), the goal is to classify proteins into one of four functional locations within the cell:

*   `Fold1`
*   `Fold2`
*   `Fold3`
*   `Fold4`

This is a multi-class classification task addressed using Python and scikit-learn.

---

## 💾 Data

The primary dataset used is `comp_occur.csv`. It contains:

*   **Labels:** The first column indicates the protein's location (`Fold1` to `Fold4`).
*   **Features:** The subsequent 40 columns represent pre-computed features based on amino acid composition and occurrence.

---

## 🛠️ Setup and Dependencies

Ensure you have Python installed. You'll also need the following core libraries:

*   **pandas:** For data manipulation and loading the CSV.
*   **numpy:** For numerical operations.
*   **scikit-learn:** For machine learning models, preprocessing, and evaluation.

You can typically install these using pip:

```bash
pip install pandas numpy scikit-learn
```

*(Optional: The script `mlproject_test_2.py` also uses `imbalanced-learn` if you plan to run variations involving SMOTE. You might need `pip install imbalanced-learn` for that.)*

---

## ▶️ Running the Analysis

The main script for training and evaluating the models is `mlproject_test_2.py`.

1.  Make sure `comp_occur.csv` is in the same directory as the script.
2.  Navigate to the project directory in your terminal.
3.  Run the script using Python:

```bash
python mlproject_test_2.py
```

This script will:

1.  Load and preprocess the data (including label encoding and feature scaling).
2.  Split the data into training (75%) and testing (25%) sets.
3.  Define pipelines for K-Nearest Neighbors (KNN), Support Vector Classifier (SVC), and Random Forest classifiers.
    *   *Note:* SVC and RandomForest use `class_weight='balanced'` to attempt to handle class imbalance.
4.  Evaluate models using 5-fold cross-validation on the training data.
5.  Evaluate models on the independent test set, printing accuracy, confusion matrices, and classification reports.

---

## 📊 Models Implemented

The following classification models are used within scikit-learn pipelines:

1.  **K-Nearest Neighbors (KNN):** `KNeighborsClassifier(n_neighbors=18, metric='minkowski', p=2.5)`
2.  **Support Vector Classifier (SVC):** `SVC(kernel='rbf', class_weight='balanced', decision_function_shape='ovo')`
3.  **Random Forest:** `RandomForestClassifier(n_estimators=10000, criterion='entropy', class_weight='balanced')`

---

## 📝 Project Structure

```
ml-project/
├── .gitignore           # Specifies intentionally untracked files
├── comp_occur.csv       # Dataset with features and labels
├── mlproject_test_2.py  # Main script for analysis (with class weights)
├── README.md            # This file
└── ... (other project files, potentially ignored scripts/reports)
```

---
