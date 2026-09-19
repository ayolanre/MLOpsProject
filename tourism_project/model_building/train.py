import os
import sys

# MLflow may print Unicode run links; keep local Windows execution UTF-8 compatible.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import pandas as pd
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import xgboost as xgb
import joblib
import mlflow

# GitHub Actions sets this value; the default supports local execution.
tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
mlflow.set_tracking_uri(tracking_uri)
mlflow.set_experiment("customer_purchase_prediction")

required_files = ["Xtrain.csv", "Xtest.csv", "ytrain.csv", "ytest.csv"]
for file_name in required_files:
    if not os.path.exists(file_name):
        raise FileNotFoundError(f"Required file missing: {file_name}")

Xtrain = pd.read_csv("Xtrain.csv")
Xtest = pd.read_csv("Xtest.csv")
ytrain = pd.read_csv("ytrain.csv").squeeze("columns")
ytest = pd.read_csv("ytest.csv").squeeze("columns")

numeric_features = [
    "Age", "NumberOfPersonVisiting", "PreferredPropertyStar", "NumberOfTrips",
    "NumberOfChildrenVisiting", "MonthlyIncome", "PitchSatisfactionScore",
    "NumberOfFollowups", "DurationOfPitch"
]
categorical_features = [
    "TypeofContact", "CityTier", "Occupation", "Gender", "MaritalStatus",
    "Passport", "OwnCar", "Designation", "ProductPitched"
]

class_counts = ytrain.value_counts()
class_weight = class_counts.get(0, 1) / class_counts.get(1, 1)
preprocessor = make_column_transformer(
    (StandardScaler(), numeric_features),
    (OneHotEncoder(handle_unknown="ignore"), categorical_features),
)
xgb_model = xgb.XGBClassifier(
    scale_pos_weight=class_weight,
    random_state=42,
    eval_metric="logloss",
)
model_pipeline = make_pipeline(preprocessor, xgb_model)

# Keep CI training time bounded while evaluating useful model complexity choices.
param_grid = {
    "xgbclassifier__n_estimators": [100, 200],
    "xgbclassifier__max_depth": [3, 5],
    "xgbclassifier__learning_rate": [0.05, 0.1],
    "xgbclassifier__reg_lambda": [1, 5],
}

with mlflow.start_run():
    grid_search = GridSearchCV(model_pipeline, param_grid, cv=5, n_jobs=-1, scoring="f1")
    grid_search.fit(Xtrain, ytrain)

    best_model = grid_search.best_estimator_
    mlflow.log_params(grid_search.best_params_)
    classification_threshold = 0.45
    mlflow.log_metric("classification_threshold", classification_threshold)

    y_pred_train = (best_model.predict_proba(Xtrain)[:, 1] >= classification_threshold).astype(int)
    y_pred_test = (best_model.predict_proba(Xtest)[:, 1] >= classification_threshold).astype(int)
    train_report = classification_report(ytrain, y_pred_train, output_dict=True, zero_division=0)
    test_report = classification_report(ytest, y_pred_test, output_dict=True, zero_division=0)
    mlflow.log_metrics({
        "train_accuracy": train_report["accuracy"],
        "train_precision": train_report["1"]["precision"],
        "train_recall": train_report["1"]["recall"],
        "train_f1": train_report["1"]["f1-score"],
        "test_accuracy": test_report["accuracy"],
        "test_precision": test_report["1"]["precision"],
        "test_recall": test_report["1"]["recall"],
        "test_f1": test_report["1"]["f1-score"],
    })

    model_path = "tourism_project/deployment/best_model_v1.pkl"
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(best_model, model_path)
    mlflow.log_artifact(model_path, artifact_path="model")
    print("Best parameters:", grid_search.best_params_)
    print("Test classification report:")
    print(classification_report(ytest, y_pred_test, zero_division=0))
    print(f"Model saved to {model_path}")
