import os
import json
import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    f1_score
)
from imblearn.over_sampling import SMOTE

# Set URI untuk MLflow Server (jika menggunakan MLflow lokal atau server remote)
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("Diabetes")

def run_modelling_tuning(cleaned_filepath="diabetes_cleaned.csv", model_output="rf_model_tuned.pkl"):
    """
    Langkah-langkah modelling:
    1. Load data hasil preprocessing
    2. Split data: train & test
    3. Oversampling (SMOTE) pada data train
    4. Tuning Hyperparameter menggunakan GridSearchCV
    5. Train Random Forest dengan parameter terbaik
    6. Evaluasi model
    7. Simpan model
    """

    # Enable MLflow autolog
    mlflow.sklearn.autolog()
    
    # 1. Load preprocessed dataset
    df = pd.read_csv(cleaned_filepath)

    # 2. Pisahkan fitur dan target
    X = df.drop("diabetes", axis=1)
    y = df["diabetes"]

    # 3. Split: 80% train, 20% test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 4. Tangani ketidakseimbangan dengan SMOTE pada data latih
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

    print("Distribusi label setelah SMOTE (training):")
    print(pd.Series(y_train_res).value_counts())

    # 5. Hyperparameter tuning menggunakan GridSearchCV
    param_grid = {
        "n_estimators": [100, 150],
        "max_depth": [None, 10],
        "min_samples_split": [2, 5],
    }
    clf = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(clf, param_grid, cv=3, n_jobs=-1, scoring="accuracy")
    grid_search.fit(X_train_res, y_train_res)
    best_model = grid_search.best_estimator_

    # 6. Prediksi dan evaluasi
    y_pred = best_model.predict(X_test)

    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)

    print("\nClassification Report:")
    cr = classification_report(y_test, y_pred, digits=4)
    print(cr)

    # Logging metrics and confusion matrix to MLflow
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, best_model.predict_proba(X_test)[:, 1])
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("f1_score", f1)
    mlflow.log_metric("roc_auc", auc)
    mlflow.log_metric("TP", cm[1][1])
    mlflow.log_metric("TN", cm[0][0])
    mlflow.log_metric("FP", cm[0][1])
    mlflow.log_metric("FN", cm[1][0])

    # 7. Simpan model
    joblib.dump(best_model, model_output)
    print(f"\nModel disimpan ke: {model_output}")

    # Log the model to MLflow
    mlflow.sklearn.log_model(best_model, "model")

if __name__ == "__main__":
    # Start the MLflow run
    with mlflow.start_run():
        run_modelling_tuning()