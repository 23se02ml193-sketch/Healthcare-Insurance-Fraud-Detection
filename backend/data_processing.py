import pandas as pd
import numpy as np
import os

# Set paths based on your VS Code screenshot
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPATIENT_PATH = os.path.join(BASE_DIR, '..', 'data', 'raw', 'Train_Inpatientdata-1542865627584.csv')
BENEFICIARY_PATH = os.path.join(BASE_DIR, '..', 'data', 'raw', 'Train_Beneficiarydata-1542865627584.csv')
LABEL_PATH = os.path.join(BASE_DIR, '..', 'data', 'raw', 'Train-1542865627584.csv')
OUTPUT_PATH = os.path.join(BASE_DIR, '..', 'data', 'processed', 'provider_features.csv')

def process_data():
    print("🚀 Starting Aggressive Feature Engineering...")
    
    train_inpatient = pd.read_csv(INPATIENT_PATH)
    beneficiary = pd.read_csv(BENEFICIARY_PATH)
    labels = pd.read_csv(LABEL_PATH)

    # --- THE FIX: Calculate Risk Score using actual Beneficiary columns ---
    # Kaggle uses 1 for Yes, 2 for No. We convert 2s to 0s for better AI math.
    chronic_cols = ['ChronicCond_Alzheimer', 'ChronicCond_Heartfailure', 'ChronicCond_KidneyDisease', 'ChronicCond_Cancer']
    for col in chronic_cols:
        beneficiary[col] = beneficiary[col].replace(2, 0)

    # RenalDiseaseIndicator is 'Y' (Yes) or '0' (No). Convert to 1 and 0.
    beneficiary['RenalDiseaseIndicator'] = beneficiary['RenalDiseaseIndicator'].replace({'Y': 1, '0': 0}).astype(float)

    # The new HighRisk_Score is the total number of severe chronic conditions the patient has
    beneficiary['HighRisk_Score'] = beneficiary[chronic_cols].sum(axis=1) + beneficiary['RenalDiseaseIndicator']
    # ----------------------------------------------------------------------

    # Merge Data
    df = pd.merge(train_inpatient, beneficiary, on='BeneID', how='inner')

    # ADVANCED AGGREGATION (The "Fraud Signal" Builder)
    provider_groups = df.groupby('Provider').agg({
        'ClaimID': 'count',
        'InscClaimAmtReimbursed': ['sum', 'mean', 'std'],
        'DeductibleAmtPaid': 'sum',
        'HighRisk_Score': 'mean',
        'NoOfMonths_PartACov': 'mean',
        'ChronicCond_Alzheimer': 'mean',
        'ChronicCond_Heartfailure': 'mean',
        'ChronicCond_KidneyDisease': 'mean',
        'ChronicCond_Cancer': 'mean',
        'IPAnnualReimbursementAmt': 'mean'
    })

    # Flatten columns to exactly 12 features
    provider_groups.columns = [
        'Total_Inpatient_Claims', 
        'Total_Claim_Amt', 
        'Avg_Claim_Amt', 
        'Claim_Amt_Std',
        'Total_Deductible_Amt', 
        'Patient_Risk_Profile',
        'Avg_PartA_Coverage',
        'Avg_Alzheimer_Risk', 
        'Avg_HeartFailure_Risk', 
        'Avg_KidneyDisease_Risk',
        'Avg_Cancer_Risk',
        'Avg_Annual_Reimbursement'
    ]

    provider_groups = provider_groups.fillna(0)
    final_df = pd.merge(provider_groups, labels, on='Provider', how='inner')
    
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    final_df.to_csv(OUTPUT_PATH, index=False)
    print(f"✅ Processed {len(final_df)} providers with 12 high-density features.")

if __name__ == "__main__":
    process_data()