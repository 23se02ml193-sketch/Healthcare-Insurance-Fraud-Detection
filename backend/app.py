from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
import pandas as pd
import numpy as np
import xgboost as xgb
import shap
import os
import sqlite3
import pickle
from datetime import datetime
from io import BytesIO

app = FastAPI(title="Healthcare Fraud Auditor API")

app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FRONTEND_DIR = os.path.join(BASE_DIR, '..', 'frontend')
MODEL_PATH = os.path.join(BASE_DIR, 'saved_models', 'xgboost_fraud_model_optimized.json')
SCALER_PATH = os.path.join(BASE_DIR, 'saved_models', 'scaler.pkl')
DB_PATH = os.path.join(BASE_DIR, 'audit_history.db')

booster = None
explainer = None
scaler = None

class SingleProvider(BaseModel):
    provider_id: str
    Total_Inpatient_Claims: float
    Total_Claim_Amt: float
    Avg_Claim_Amt: float
    Claim_Amt_Std: float
    Total_Deductible_Amt: float
    Patient_Risk_Profile: float
    Avg_PartA_Coverage: float
    Avg_Alzheimer_Risk: float
    Avg_HeartFailure_Risk: float
    Avg_KidneyDisease_Risk: float
    Avg_Cancer_Risk: float
    Avg_Annual_Reimbursement: float
    threshold: float = 0.35

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS single_audits (id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp TEXT, provider_id TEXT, risk_score REAL, fraud_flag TEXT, reason TEXT)''')
    conn.commit()
    conn.close()

def save_batch_to_db(records):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    data_to_insert = [(timestamp, r["provider_id"], r["risk_score"], r["fraud_flag"], r["reason"]) for r in records]
    c.executemany('''INSERT INTO single_audits (timestamp, provider_id, risk_score, fraud_flag, reason) VALUES (?, ?, ?, ?, ?)''', data_to_insert)
    conn.commit()
    conn.close()

@app.on_event("startup")
def load_model():
    global booster, explainer, scaler
    init_db() 
    if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
        booster = xgb.Booster()
        booster.load_model(MODEL_PATH)
        explainer = shap.TreeExplainer(booster)
        with open(SCALER_PATH, 'rb') as f:
            scaler = pickle.load(f)
        print("✅ XGBoost, SHAP, Scaler, and SQLite Database loaded.")
    else:
        print("❌ Warning: Model or Scaler file not found.")

def analyze_risk_and_explain(df, threshold=0.35):
    X_scaled = scaler.transform(df)
    dmatrix = xgb.DMatrix(X_scaled)
    probabilities = booster.predict(dmatrix) 
    
    audit_lexicon = {
        'Total_Claim_Amt': "Overall billed volume exceeds benchmarks",
        'Total_Deductible_Amt': "Irregular patient deductible accumulation",
        'Avg_Claim_Amt': "Average cost per claim is unusually high",
        'Claim_Amt_Std': "High variance indicates inconsistent coding",
        'Total_Inpatient_Claims': "Suspicious frequency of admissions",
        'Patient_Risk_Profile': "Anomalous concentration of high-risk patients",
        'Avg_PartA_Coverage': "Irregularities in Part A Medicare durations",
        'Avg_Alzheimer_Risk': "Disproportionate billing for Alzheimer's care",
        'Avg_HeartFailure_Risk': "Anomalous claim volume for Heart Failure",
        'Avg_KidneyDisease_Risk': "Irregular billing patterns for Renal/ESRD",
        'Avg_Cancer_Risk': "Unusual frequency of Oncology claims",
        'Avg_Annual_Reimbursement': "Annual reimbursement rates exceed thresholds"
    }
    
    # These are the generic volume metrics we want to limit
    financial_cols = ['Total_Claim_Amt', 'Total_Deductible_Amt', 'Total_Inpatient_Claims', 'Avg_Claim_Amt']
    
    results = []
    try:
        shap_vals = explainer.shap_values(X_scaled) 
        
        for i in range(len(df)):
            prob = float(probabilities[i])
            risk_score = float(round(prob * 100, 2))
            is_fraud = "Yes" if prob >= threshold else "No" 
            
            reason = "Standard billing profile within normal limits."
            if is_fraud == "Yes":
                feature_contributions = shap_vals[i]
                top_indices = np.argsort(feature_contributions)[::-1] 
                
                driving_factors = []
                fin_count = 0
                
                # Loop through all features ordered by SHAP importance
                for idx in top_indices: 
                    shap_val = feature_contributions[idx]
                    
                    # Lowered threshold to 0.01 to allow nuanced clinical factors to appear
                    if shap_val > 0.01: 
                        col_name = df.columns[idx]
                        
                        # LIMIT THE SPAM: Only allow 1 generic financial reason
                        if col_name in financial_cols:
                            fin_count += 1
                            if fin_count > 1:
                                continue # Skip to find a more interesting clinical reason
                        
                        # INJECT REAL VALUES: Get the actual number to make it unique
                        actual_value = df.iloc[i][col_name]
                        if 'Amt' in col_name or 'Reimbursement' in col_name:
                            val_str = f"${actual_value:,.0f}"
                        elif 'Risk' in col_name or 'Profile' in col_name or 'Coverage' in col_name:
                            val_str = f"Score: {actual_value:.2f}"
                        else:
                            val_str = f"{actual_value:,.0f}"
                            
                        human_text = audit_lexicon.get(col_name, f"Anomalous {col_name}")
                        driving_factors.append(f"• {human_text} <b>[{val_str}]</b>")
                        
                        # Stop once we have 3 good reasons
                        if len(driving_factors) >= 3:
                            break
                
                if len(driving_factors) > 0:
                    reason = "<br>".join(driving_factors)
                else:
                    reason = "• Complex multi-factor risk pattern identified."
                    
            elif risk_score < 20:
                reason = "Very low risk indicators. Verified clean."

            results.append({"fraud_flag": is_fraud, "risk_score": risk_score, "reason": reason})
    except Exception as e:
        print(f"SHAP Error: {e}")
        for i in range(len(df)):
            prob = float(probabilities[i])
            results.append({"fraud_flag": "Yes" if prob >= threshold else "No", "risk_score": float(round(prob * 100, 2)), "reason": "AI mathematical explanation unavailable."})
    return results

@app.post("/api/audit-single")
async def audit_single(data: SingleProvider):
    if booster is None: raise HTTPException(status_code=500, detail="AI Model not loaded.")
    try:
        data_dict = data.dict()
        provider_id = data_dict.pop('provider_id')
        custom_threshold = data_dict.pop('threshold') 
        
        expected_features = [
            'Total_Inpatient_Claims', 'Total_Claim_Amt', 'Avg_Claim_Amt', 'Claim_Amt_Std',
            'Total_Deductible_Amt', 'Patient_Risk_Profile', 'Avg_PartA_Coverage',
            'Avg_Alzheimer_Risk', 'Avg_HeartFailure_Risk', 'Avg_KidneyDisease_Risk',
            'Avg_Cancer_Risk', 'Avg_Annual_Reimbursement'
        ]
        
        df_features = pd.DataFrame([[data_dict[f] for f in expected_features]], columns=expected_features)
        
        analysis = analyze_risk_and_explain(df_features, threshold=custom_threshold)[0]
        analysis["provider_id"] = provider_id
        
        return {"status": "success", "data": [analysis]}
    except Exception as e: raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/audit-providers")
async def audit_providers(file: UploadFile = File(...), threshold: float = Form(0.35)):
    if booster is None: raise HTTPException(status_code=500, detail="AI Model not loaded.")
    if not file.filename.endswith('.csv'): raise HTTPException(status_code=400, detail="Only CSV files accepted.")
    
    try:
        contents = await file.read()
        df = pd.read_csv(BytesIO(contents))
    except: raise HTTPException(status_code=400, detail="Could not read CSV.")
        
    if 'Provider' not in df.columns: raise HTTPException(status_code=400, detail="Missing 'Provider' column.")
    
    expected_features = [
        'Total_Inpatient_Claims', 'Total_Claim_Amt', 'Avg_Claim_Amt', 'Claim_Amt_Std',
        'Total_Deductible_Amt', 'Patient_Risk_Profile', 'Avg_PartA_Coverage',
        'Avg_Alzheimer_Risk', 'Avg_HeartFailure_Risk', 'Avg_KidneyDisease_Risk',
        'Avg_Cancer_Risk', 'Avg_Annual_Reimbursement'
    ]
    
    missing = [f for f in expected_features if f not in df.columns]
    if missing: raise HTTPException(status_code=400, detail=f"Missing columns: {', '.join(missing)}")
        
    try:    
        provider_ids = df['Provider']
        X_audit = df[expected_features]
        
        raw_results = analyze_risk_and_explain(X_audit, threshold=threshold)
        
        final_results = []
        for i in range(len(provider_ids)):
            res = raw_results[i]
            res["provider_id"] = str(provider_ids.iloc[i])
            final_results.append(res)
            
        final_results.sort(key=lambda x: x['risk_score'], reverse=True)
        save_batch_to_db(final_results)
        return {"status": "success", "total_audited": len(final_results), "data": final_results}
    except Exception as e: raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/history")
def get_history():
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        c = conn.cursor()
        c.execute('SELECT * FROM single_audits ORDER BY id DESC LIMIT 100')
        rows = c.fetchall()
        conn.close()
        return {"status": "success", "data": [dict(row) for row in rows]}
    except Exception as e: return {"status": "error", "detail": str(e)}

@app.get("/")
def serve_dashboard():
    return FileResponse(os.path.join(FRONTEND_DIR, 'index.html'))