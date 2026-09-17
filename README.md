# MLE_Case_Study_CA1
Hospital 30-Day Readmission Prediction

📌 Project Overview

This project implements a Hospital 30-Day Readmission Prediction
case study using Logistic Regression.

The supplied hospital dataset contains a readmission_rate column
rather than a patient-level binary 30-day readmission outcome.
Therefore, the notebook creates a binary target named readmission_30d
using the median readmission rate as the cutoff.

The project then uses hospital-related numerical features to classify
observations into two groups:

0 → Lower readmission-risk group

1 → Higher readmission-risk group

Important: readmission_30d is a constructed classification
target based on the dataset median. It is not an actual patient-level
30-day readmission label.

🎯 Objectives

Load and inspect the hospital dataset.

Create a binary readmission-risk target.

Separate input features (X) and target (y).

Split the data into training and testing sets.

Handle missing numerical values.

Standardize numerical features.

Train a Logistic Regression classifier with L2 regularization.

Evaluate the model using ROC-AUC, precision, recall, confusion
matrix, and classification report.

Inspect Logistic Regression coefficients to understand feature
associations.

📂 Dataset

The notebook downloads the dataset using KaggleHub:

kagglehub.dataset_download("vipulshahi/patient-readdmission-dataset")

The dataset is loaded into a Pandas DataFrame using:

pd.read_csv()

Main columns

Column                            Description

date                            Date associated with the hospital data
num_patients_admitted           Number of patients admitted
avg_length_of_stay              Average length of hospital stay
avg_lab_result_score            Average laboratory result score
hospital_resource_utilization   Hospital resource utilization
readmission_rate                Readmission rate used to construct the target

🧠 Methodology

1. Create the Target Variable

The notebook creates readmission_30d using the median of
readmission_rate:

df["readmission_30d"] = (
    df["readmission_rate"] >= df["readmission_rate"].median()
).astype(int)

The comparison produces True or False, which is converted to:

True  → 1
False → 0

2. Define Features and Target

The following columns are removed from the model inputs:

readmission_rate --- used to create the target

readmission_30d --- the target itself

date --- not used as a model feature

X = df.drop(
    columns=["readmission_rate", "readmission_30d", "date"],
    errors="ignore"
)

y = df["readmission_30d"]

The model features are therefore the available numerical hospital
variables.

3. Train-Test Split

The data is divided into:

80% training data

20% testing data

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=.2,
    stratify=y,
    random_state=42
)

stratify=y helps preserve the class distribution in the training and
testing sets.

random_state=42 makes the split reproducible.

4. Data Preprocessing

The notebook uses a preprocessing pipeline containing two steps:

Missing-value handling

SimpleImputer(strategy="median")

Missing numerical values are replaced with the median of the
corresponding feature.

Feature scaling

StandardScaler()

The numerical features are standardized before they are given to the
Logistic Regression model.

These steps are combined using Pipeline and ColumnTransformer.

🤖 Machine Learning Model

Logistic Regression

The main classification algorithm is:

LogisticRegression(
    C=1,
    penalty="l2",
    solver="liblinear",
    max_iter=2000
)

Parameters

Parameter            Value Purpose

C                    1 Controls the inverse strength of regularization
penalty             l2 Uses L2 regularization
solver       liblinear Optimization algorithm used for training
max_iter          2000 Maximum number of training iterations

The preprocessing and model are combined into one pipeline:

Input Data
    ↓
Missing Value Imputation
    ↓
Standardization
    ↓
Logistic Regression
    ↓
Prediction

🔮 Prediction

The trained model generates class probabilities using:

p = model.predict_proba(X_test)[:, 1]

[:, 1] selects the probability of class 1.

The probability is converted into a class prediction using a threshold
of 0.5:

pred = (p >= .5).astype(int)

Therefore:

Probability >= 0.5 → Class 1
Probability <  0.5 → Class 0

📊 Model Evaluation

The notebook evaluates the model using several metrics.

ROC-AUC

roc_auc_score(y_test, p)

ROC-AUC measures how well the model separates the two classes across
different probability thresholds.

The notebook also performs 5-fold stratified cross-validation:

StratifiedKFold(
    5,
    shuffle=True,
    random_state=42
)

with:

cross_val_score(..., scoring="roc_auc")

Precision

precision_score(y_test, pred)

Precision answers:

Of the observations predicted as class 1, how many were actually class
1?

Formula:

Precision = TP / (TP + FP)

Recall

recall_score(y_test, pred)

Recall answers:

Of all actual class-1 observations, how many were correctly
identified?

Formula:

Recall = TP / (TP + FN)

Confusion Matrix

confusion_matrix(y_test, pred)

The confusion matrix contains:

                       Predicted 0           Predicted 1

Actual 0      True Negative (TN)   False Positive (FP)
Actual 1     False Negative (FN)    True Positive (TP)

The notebook also visualizes the confusion matrix using Matplotlib.

Classification Report

classification_report(
    y_test,
    pred,
    digits=4
)

This reports:

Precision

Recall

F1-score

Support

for each class.

🔍 Feature Coefficients

The notebook extracts the Logistic Regression coefficients:

coef = pd.Series(
    model.named_steps["lr"].coef_[0],
    index=num
).sort_values(
    key=np.abs,
    ascending=False
)

The coefficients are sorted by their absolute magnitude.

Because the numerical features are standardized before Logistic
Regression, the coefficient magnitudes can be compared more meaningfully
across the features.

A positive coefficient indicates an association with a higher model
log-odds for class 1, while a negative coefficient indicates an
association with lower log-odds for class 1.

These coefficients show model associations; they should not be
interpreted as proof of causation.

🛠️ Technologies Used

Python

Google Colab

Pandas

NumPy

Matplotlib

Scikit-learn

KaggleHub

📚 Libraries Used

import os
import glob
import kagglehub
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_val_score
)

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    roc_auc_score,
    confusion_matrix,
    precision_score,
    recall_score,
    classification_report
)

▶️ How to Run the Project

Option 1: Google Colab

Open the .ipynb notebook in Google Colab.

Run the cells from top to bottom.

The notebook downloads the dataset through KaggleHub.

The data is loaded and preprocessed.

Logistic Regression is trained.

Evaluation metrics are displayed.

The confusion matrix is plotted.

Feature coefficients are displayed.

Option 2: GitHub

Upload these files to your repository:

README.md
unit1_hospital_readmission_case_study_updated_(1)(1).ipynb

The notebook can then be opened directly from the GitHub repository.

📁 Project Structure

Hospital-Readmission-Prediction/
│
├── README.md
│
└── unit1_hospital_readmission_case_study_updated_(1)(1).ipynb

🔄 Project Workflow

Hospital Dataset
       ↓
Load CSV
       ↓
Data Inspection
       ↓
Create Binary Target
       ↓
Separate X and y
       ↓
Train-Test Split
       ↓
Median Imputation
       ↓
Standard Scaling
       ↓
Logistic Regression
       ↓
Probability Prediction
       ↓
0.5 Threshold
       ↓
Class Prediction
       ↓
Model Evaluation
       ↓
ROC-AUC
Confusion Matrix
Precision
Recall
F1-score
       ↓
Feature Coefficients

⚠️ Limitations

The supplied dataset contains readmission_rate, not an actual
patient-level 30-day readmission outcome.

The binary target readmission_30d is created using the median
readmission rate as a cutoff.

Therefore, the result should be understood as a readmission-risk
classification exercise, rather than a validated clinical
prediction system.

The model is intended for this academic case study and should not be
used for real clinical decision-making without appropriate
patient-level data, validation, and clinical evaluation.

🎓 Case Study Summary

This project demonstrates how Logistic Regression can be used for a
binary classification task related to hospital readmission risk.

The workflow covers the complete machine-learning process:

Data Loading → Target Creation → Preprocessing → Train/Test Split →
Logistic Regression → Prediction → Evaluation → Feature Interpretation

👨‍💻 Project Type

Academic Case Study - Machine Learning Practice

Topic: Hospital 30-Day Readmission Prediction
Algorithm: Logistic Regression
Regularization: L2
Platform: Google Colab
