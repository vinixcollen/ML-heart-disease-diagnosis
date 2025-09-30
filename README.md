# Heart Disease Diagnosis using Random Forest

A machine learning system for binary classification of heart disease presence using clinical patient data. Implements Random Forest with hyperparameter optimization to achieve 92.9% recall and 86.7% precision on test data.

## 📋 Project Overview

This project develops a clinical screening tool for heart disease detection using machine learning. The system utilizes 13 routinely collected clinical parameters to identify patients requiring cardiology referral, with a focus on minimizing missed diagnoses through systematic model optimization.

## 🎯 Clinical Motivation

Cardiovascular diseases are the leading cause of death worldwide. Early detection is crucial for effective treatment, but many healthcare systems lack accessible screening tools. This project addresses this gap by creating a reliable, ML-based screening assistant for primary care settings.

## 📊 Dataset

**Source**: UCI Heart Disease Dataset (Cleveland)
- **Patients**: 303 records
- **Features**: 13 clinical parameters
- **Original Target**: 5-class severity (0-4)
- **Updated Target**: Binary classification (0: No disease, 1: Disease present)
- **Class Distribution**: 164 healthy (54.1%), 139 diseased (45.9%)

**Clinical Features**:
- Demographic: age, sex
- Clinical measurements: resting BP, cholesterol, fasting blood sugar
- ECG results: resting electrocardiographic
- Exercise-related: max heart rate, exercise-induced angina
- ST analysis: ST depression, slope
- Angiographic: vessels colored, thallium stress test
- Symptomatic: chest pain type

## 🛠️ Implementation

### Data Preprocessing
- Missing value imputation using mode from training set
- Label encoding for categorical variables
- 70-20-10 stratified train-validation-test split
- Class balancing for 45.9% disease prevalence

### Model Architecture
```python
RandomForestClassifier(
    n_estimators=100,
    max_depth=5,  # Optimized from initial 10
    min_samples_split=5,
    min_samples_leaf=2,
    max_features='sqrt',
    class_weight='balanced',
    random_state=42
)
