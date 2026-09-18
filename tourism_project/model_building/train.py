# for data manipulation
import os
import pandas as pd
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import joblib
import mlflow

# ----------------------------
# MLflow Setup
# ----------------------------
mlflow.set_tracking_uri("http://127.0.0.1:5000")
# Connects to the MLflow tracking server running locally or in GitHub Actions
mlflow.set_experiment("customer_purchase_prediction")
# Creates or selects the experiment where all runs will be logged

# ----------------------------
# Load Data
# ----------------------------
required_files = ["Xtrain.csv", "Xtest.csv", "ytrain.csv", "ytest.csv"]
# Ensures all required split files exist before training
for f in required_files:
    if not os.path.exists(f):
        raise FileNotFoundError(f"Required file missing: {f}")

# Load training and testing data
Xtrain = pd.read_csv("Xtrain.csv")
Xtest = pd.read_csv("Xtest.csv")
ytrain = pd.read_csv("ytrain.csv").squeeze()
ytest = pd.read_csv("ytest.csv").squeeze()

# ----------------------------
# Feature Groups
# ----------------------------
# Numeric features to be scaled
numeric_features = [
    "Age",
    "NumberOfPersonVisiting",
    "PreferredPropertyStar",
    "NumberOfTrips",
    "NumberOfChildrenVisiting",
    "MonthlyIncome",
    "PitchSatisfactionScore",
    "NumberOfFollowups",
    "DurationOfPitch"
]

# Categorical features to be one‑hot encoded
categorical_features = [
    "TypeofContact",
    "CityTier",
    "Occupation",
    "Gender",
    "MaritalStatus",
    "Passport",
    "OwnCar",
    "Designation",
    "ProductPitched"
]

# ----------------------------
# Handle Class Imbalance
# ----------------------------
# Compute ratio of majority/minority class for XGBoost weighting
class_counts = ytrain.value_counts()
class_weight = class_counts.get(0, 1) / class_counts.get(1, 1)

# ----------------------------
# Preprocessing Pipeline
# ----------------------------
# Scale numeric features + one‑hot encode categorical features
preprocessor = make_column_transformer(
    (StandardScaler(), numeric_features),
    (OneHotEncoder(handle_unknown="ignore"), categorical_features)
)

# ----------------------------
# Base Model
# ----------------------------
# XGBoost classifier with class imbalance handling
xgb_model = xgb.XGBClassifier(
    scale_pos_weight=class_weight,
    random_state=42,
    eval_metric="logloss"
)

# Combine preprocessing + model into a single pipeline
model_pipeline = make_pipeline(preprocessor, xgb_model)

# ----------------------------
# Hyperparameter Grid
# ----------------------------
# Search space for GridSearchCV
param_grid = {
    "xgbclassifier__n_estimators": [100, 200, 300],
    "xgbclassifier__max_depth": [3, 5, 7],
    "xgbclassifier__colsample_bytree": [0.6, 0.8, 1.0],
    "xgbclassifier__colsample_bylevel": [0.6, 0.8, 1.0],
    "xgbclassifier__learning_rate": [0.01, 0.05, 0.1],
    "xgbclassifier__reg_lambda": [1, 2, 5],
}

# ----------------------------
# MLflow Run
# ----------------------------
with mlflow.start_run():

    # Perform hyperparameter tuning
    grid_search = GridSearchCV(model_pipeline, param_grid, cv=5, n_jobs=-1)
    grid_search.fit(Xtrain, ytrain)

    # Log every hyperparameter combination tested
    results = grid_search.cv_results_
    for i, params in enumerate(results["params"]):
        with mlflow.start_run(nested=True):
            mlflow.log_params(params)
            mlflow.log_metric("mean_test_score", results["mean_test_score"][i])
            mlflow.log_metric("std_test_score", results["std_test_score"][i])

    # Retrieve best model and log its parameters
    best_model = grid_search.best_estimator_
    mlflow.log_params(grid_search.best_params_)

    # Custom classification threshold
    classification_threshold = 0.45
    mlflow.log_metric("classification_threshold", classification_threshold)

    # Predictions for train and test sets
    y_pred_train_proba = best_model.predict_proba(Xtrain)[:, 1]
    y_pred_train = (y_pred_train_proba >= classification_threshold).astype(int)

    y_pred_test_proba = best_model.predict_proba(Xtest)[:, 1]
    y_pred_test = (y_pred_test_proba >= classification_threshold).astype(int)

    # Generate evaluation metrics
    train_report = classification_report(ytrain, y_pred_train, output_dict=True)
    test_report = classification_report(ytest, y_pred_test, output_dict=True)

    # Log evaluation metrics to MLflow
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

    # ----------------------------
    # Save Model
    # ----------------------------
    model_path = "tourism_project/deployment/best_model_v1.pkl"
    # Save trained model to disk
    joblib.dump(best_model, model_path)
    # Log model artifact to MLflow
    mlflow.log_artifact(model_path, artifact_path="model")

    print(f"Model saved to {model_path}")
