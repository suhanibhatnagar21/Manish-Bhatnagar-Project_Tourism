import os
import pickle
import warnings
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    precision_score, recall_score, classification_report
)
from huggingface_hub import HfApi, login

warnings.filterwarnings('ignore')

# ── Authenticate ──────────────────────────────────────────────────────────
HF_TOKEN = os.environ.get('HF_TOKEN')
login(token=HF_TOKEN)

DATASET_BASE = 'hf://datasets/Suhani2128/tourism-dataset'

# ── Load train / test data from the Hugging Face data space ─────────────
print('Loading train/test data from Hugging Face...')
train_df = pd.read_csv(f'{DATASET_BASE}/train.csv')
test_df  = pd.read_csv(f'{DATASET_BASE}/test.csv')
print(f'Train: {train_df.shape}  |  Test: {test_df.shape}')

TARGET  = 'ProdTaken'
X_train = train_df.drop(TARGET, axis=1)
y_train = train_df[TARGET]
X_test  = test_df.drop(TARGET, axis=1)
y_test  = test_df[TARGET]

# ── Define model and hyperparameter grid ──────────────────────────────────
# Use scale_pos_weight to handle class imbalance
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

xgb_model = XGBClassifier(
    eval_metric       = 'logloss',
    random_state      = 42,
    scale_pos_weight  = scale_pos_weight,
    use_label_encoder = False
)

param_grid = {
    'n_estimators'    : [100, 200, 300],
    'max_depth'       : [3, 5, 7],
    'learning_rate'   : [0.05, 0.1, 0.2],
    'subsample'       : [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0],
}

# ── Tune the model with GridSearchCV (3-fold stratified CV) ───────────────
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
print('Starting GridSearchCV hyperparameter tuning...')
gs = GridSearchCV(
    estimator  = xgb_model,
    param_grid = param_grid,
    cv         = cv,
    scoring    = 'f1',
    n_jobs     = -1,
    refit      = True,
    verbose    = 1
)
gs.fit(X_train, y_train)
best_model = gs.best_estimator_
print(f'Best params: {gs.best_params_}')

# ── Evaluate model performance on the test set ────────────────────────────
y_pred      = best_model.predict(X_test)
y_pred_prob = best_model.predict_proba(X_test)[:, 1]

acc  = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec  = recall_score(y_test, y_pred)
f1   = f1_score(y_test, y_pred)
auc  = roc_auc_score(y_test, y_pred_prob)

print('\n=== Model Performance on Test Set ===')
print(f'  Accuracy  : {acc:.4f}')
print(f'  Precision : {prec:.4f}')
print(f'  Recall    : {rec:.4f}')
print(f'  F1-Score  : {f1:.4f}')
print(f'  ROC-AUC   : {auc:.4f}')
print(classification_report(y_test, y_pred, target_names=['No Purchase','Purchase']))

# ── Log all tuned parameters and metrics with MLflow ──────────────────────
mlflow.set_tracking_uri('mlruns')
mlflow.set_experiment('Tourism_Package_Prediction')

with mlflow.start_run(run_name='XGBoost_GridSearchCV'):
    # Log all hyperparameters from GridSearchCV
    mlflow.log_params(gs.best_params_)
    mlflow.log_metric('cv_best_f1'    , round(gs.best_score_, 4))
    mlflow.log_metric('test_accuracy' , round(acc,  4))
    mlflow.log_metric('test_precision', round(prec, 4))
    mlflow.log_metric('test_recall'   , round(rec,  4))
    mlflow.log_metric('test_f1_score' , round(f1,   4))
    mlflow.log_metric('test_roc_auc'  , round(auc,  4))
    mlflow.sklearn.log_model(best_model, artifact_path='xgboost_model')
    run_id = mlflow.active_run().info.run_id
    print(f'MLflow run ID: {run_id}')

# ── Save the best model locally ───────────────────────────────────────────
os.makedirs('tourism_project/model_building', exist_ok=True)
MODEL_PATH = 'tourism_project/model_building/best_model.pkl'
with open(MODEL_PATH, 'wb') as f:
    pickle.dump(best_model, f)
print(f'Model saved locally → {MODEL_PATH}')

# ── Register the best model in the Hugging Face model hub ─────────────────
api     = HfApi()
repo_id = 'Suhani2128/tourism-prediction-model'

api.create_repo(
    repo_id   = repo_id,
    repo_type = 'model',
    exist_ok  = True,
    token     = HF_TOKEN
)
api.upload_file(
    path_or_fileobj = MODEL_PATH,
    path_in_repo    = 'best_model.pkl',
    repo_id         = repo_id,
    repo_type       = 'model',
    token           = HF_TOKEN
)
print(f'Model registered at: https://huggingface.co/{repo_id}')
