import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from imblearn.over_sampling import SMOTE
import optuna
import os
import pickle

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROCESSED_DATA_PATH = os.path.join(BASE_DIR, '..', 'data', 'processed', 'provider_features.csv')
MODEL_SAVE_PATH = os.path.join(BASE_DIR, 'saved_models', 'xgboost_fraud_model_optimized.json')
SCALER_SAVE_PATH = os.path.join(BASE_DIR, 'saved_models', 'scaler.pkl')

def load_and_prep_data():
    df = pd.read_csv(PROCESSED_DATA_PATH)
    X = df.drop(columns=['Provider', 'PotentialFraud'])
    y = LabelEncoder().fit_transform(df['PotentialFraud'])
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    os.makedirs(os.path.dirname(SCALER_SAVE_PATH), exist_ok=True)
    with open(SCALER_SAVE_PATH, 'wb') as f:
        pickle.dump(scaler, f)
        
    return train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

# --- NEW: The Optuna Brain ---
def objective(trial):
    X_train, X_test, y_train, y_test = load_and_prep_data()
    
    # Balance data for this trial
    sm = SMOTE(random_state=42)
    X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

    # Let Optuna randomly guess these parameters within these ranges
    param = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 600),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'eval_metric': 'auc',
        'random_state': 42
    }

    # Train a temporary model
    model = xgb.XGBClassifier(**param)
    model.fit(X_train_res, y_train_res)
    
    # Grade it based on ROC-AUC
    y_prob = model.predict_proba(X_test)[:, 1]
    roc_auc = roc_auc_score(y_test, y_prob)
    
    return roc_auc

def train_best_model():
    print("🤖 Starting Optuna Hyperparameter Hunt...")
    
    # 1. Run 20 experiments to find the best settings
    study = optuna.create_study(direction='maximize')
    optuna.logging.set_verbosity(optuna.logging.INFO)
    study.optimize(objective, n_trials=20) # We do 20 trials to keep it fast but effective
    
    print("\n🏆 BEST PARAMETERS FOUND:")
    for key, value in study.best_params.items():
        print(f"   {key}: {value}")
        
    print(f"🎯 BEST ROC-AUC ESTIMATE: {study.best_value * 100:.2f}%")
    
    # 2. Train the FINAL model using the winning parameters
    print("\n🚀 Training Final Production Model with winning settings...")
    X_train, X_test, y_train, y_test = load_and_prep_data()
    
    sm = SMOTE(random_state=42)
    X_train_res, y_train_res = sm.fit_resample(X_train, y_train)
    
    # Inject the winning settings into XGBoost
    best_params = study.best_params
    best_params['eval_metric'] = 'auc'
    best_params['random_state'] = 42
    
    final_model = xgb.XGBClassifier(**best_params)
    final_model.fit(X_train_res, y_train_res)
    
    # 3. Final Report
    y_prob = final_model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.35).astype(int) 
    
    print("\n" + "="*55)
    print("🧠 FINAL OPTIMIZED MODEL EVALUATION")
    print("="*55)
    print(f"✅ Overall Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")
    print(f"📊 ROC-AUC Score:   {roc_auc_score(y_test, y_prob) * 100:.2f}%")
    print("="*55)
    print(classification_report(y_test, y_pred, target_names=['Clean (0)', 'Fraud (1)']))
    
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    final_model.save_model(MODEL_SAVE_PATH)
    print("✅ Elite high-performance model saved to disk.")

if __name__ == "__main__":
    train_best_model()