"""
End-to-end reproducible pipeline for the diabetes readmission project.

Loads and preprocesses the data, fits final models (logistic regression, Random Forest, XGBoost) 
using the hyperparameters selected during tuning (which occurs 
in the notebooks 02_baseline_model.ipynb, 03_random_forest.ipynb, 04_xgboost.ipynb), evaluates
them on the held-out test set, and saves the fitted models to disk.

Excludes hyperparameter tuning and threshold selection, which are performed in the notebooks. 
Also excludes plots and visualizations, which are generated in the notebooks.
"""

import joblib
import pandas as pd
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from src.data_loader import load_features
from src.preprocessing import clean_data, encode_categoricals
from src.split import patient_split, get_features_and_target
from src.evaluate import compute_metrics, evaluate_model


MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)


def main():
    # Load, clean, split, encode data (derived from src/ functions)
    print("Loading and preprocessing data...")
    df = load_features()
    df = clean_data(df)

    train_df, test_df = patient_split(df)

    X_train, y_train, ids_train = get_features_and_target(train_df)
    X_test, y_test, ids_test = get_features_and_target(test_df)

    X_train, X_test = encode_categoricals(X_train, X_test)

    #  Logistic Regression: scale, fit final model 
    print("\nFitting final Logistic Regression model...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    final_lr = LogisticRegression(
        C=0.01,
        l1_ratio=1,          # equivalent to penalty='l1'
        solver='liblinear',
        class_weight='balanced',
        max_iter=1000,
        random_state=42,
    )
    final_lr.fit(X_train_scaled, y_train)

    lr_train_metrics = compute_metrics(y_train, final_lr.predict_proba(X_train_scaled))
    lr_test_metrics = compute_metrics(y_test, final_lr.predict_proba(X_test_scaled))
    print(pd.DataFrame([lr_train_metrics, lr_test_metrics], index=["train", "test"]))

    #  Random Forest: fit final model 
    print("\nFitting final Random Forest model...")
    final_rf = RandomForestClassifier(
        max_depth=9,
        min_samples_leaf=20,
        max_features=0.3,
        n_estimators=400,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1,
    )
    final_rf.fit(X_train, y_train)

    rf_train_metrics = compute_metrics(y_train, final_rf.predict_proba(X_train))
    rf_test_metrics = compute_metrics(y_test, final_rf.predict_proba(X_test))
    print(pd.DataFrame([rf_train_metrics, rf_test_metrics], index=["train", "test"]))
    
    #  XGBoost: fit final model 
    print("\nFitting final XGBoost model...")
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

    final_xgb = XGBClassifier(
        max_depth=5,
        learning_rate=0.01,
        n_estimators=700,
        subsample=0.7,
        colsample_bytree=0.8,
        min_child_weight=15,
        scale_pos_weight=scale_pos_weight,
        eval_metric='aucpr',
        random_state=42,
        n_jobs=-1,
    )
    final_xgb.fit(X_train, y_train)

    xgb_train_metrics = compute_metrics(y_train, final_xgb.predict_proba(X_train))
    xgb_test_metrics = compute_metrics(y_test, final_xgb.predict_proba(X_test))
    print(pd.DataFrame([xgb_train_metrics, xgb_test_metrics], index=["train", "test"]))

    # Save models
    print("\nSaving models...")
    joblib.dump(final_lr, MODELS_DIR / "logistic_regression_final.joblib")
    joblib.dump(scaler, MODELS_DIR / "scaler.joblib")
    joblib.dump(final_rf, MODELS_DIR / "random_forest_final.joblib")
    joblib.dump(final_xgb, MODELS_DIR / "xgboost_final.joblib")
    print(f"Models saved to {MODELS_DIR.resolve()}")


if __name__ == "__main__":
    main()