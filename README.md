# 🧩 Student Depression Classification Models

This repository contains a comprehensive **machine learning classification pipeline** built to predict **student depression levels** based on demographic, lifestyle, and academic data. The notebook explores multiple models from **scikit-learn**, **XGBoost**, and **CatBoost**, with performance evaluation through cross-validation and ROC–AUC analysis.

---

## 📘 Project Overview

The goal of this project is to classify whether a student is likely to experience depression based on features such as:
- **Degree**, **City**, **Profession**
- **Sleep Duration**
- **Lifestyle and personal habits**

The dataset used is:  
`data/student_depression_dataset.csv`

The workflow includes:
1. **Data preprocessing and cleaning**
2. **Encoding categorical variables**
3. **Feature scaling for numeric data**
4. **Model training and comparison**
5. **Performance evaluation using key classification metrics**

---

## 🧹 Data Preprocessing

Key steps performed:
- Replaced rare degree categories (occurrences < 500) with `"Other"`
- Handled categorical encoding with `LabelEncoder`
- Standardized numerical features using `StandardScaler`
- Balanced class distribution via dataset manipulation
- Split data into **training and testing sets** using `train_test_split`

---

## 🧠 Models Implemented

| Model | Library | Description |
|--------|----------|-------------|
| **Logistic Regression** | `sklearn.linear_model` | Baseline linear model for binary classification. |
| **Decision Tree Classifier** | `sklearn.tree` | Non-linear model for interpretable decision boundaries. |
| **Random Forest Classifier** | `sklearn.ensemble` | Ensemble of decision trees reducing overfitting. |
| **K-Nearest Neighbors (KNN)** | `sklearn.neighbors` | Distance-based classifier using nearest samples. |
| **Support Vector Machine (SVC)** | `sklearn.svm` | Finds optimal hyperplane for separation; kernel-based. |
| **XGBoost Classifier** | `xgboost` | Gradient boosting model optimized for performance. |
| **CatBoost Classifier** | `catboost` | Boosting algorithm handling categorical data efficiently. |

---

## ⚙️ Model Training and Evaluation

### Cross Validation
Each model was trained using **K-Fold Cross Validation** to ensure robustness and prevent overfitting.

### Hyperparameter Tuning
`GridSearchCV` was used to optimize model parameters for the best validation performance.

### Evaluation Metrics
Performance metrics include:
- **Accuracy**
- **Precision**
- **Recall**
- **F1 Score**
- **ROC–AUC Score**

### Visualization
- **Confusion Matrix Display** to visualize classification accuracy per class.  
- **ROC Curve Display** to compare models’ trade-offs between sensitivity and specificity.

---

## 📊 Results

Each model’s performance was compared based on **accuracy** and **ROC–AUC**.  
Ensemble methods (Random Forest, XGBoost, CatBoost) generally provided the best balance between recall and precision.

---

**Main Libraries:**
- `scikit-learn`
- `xgboost`
- `catboost`
- `pandas`, `numpy`
- `matplotlib`, `plotly`
