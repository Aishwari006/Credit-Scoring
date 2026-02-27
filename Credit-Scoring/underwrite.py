import joblib
import pandas as pd
import os
from behaviour_engine import behavioural_features, load_transactions
from explainability import generate_explanation
import numpy as np


# set absolute paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "behaviour_risk_model.pkl")
FEATURES_PATH = os.path.join(BASE_DIR, "models", "behaviour_feature_columns.pkl")

MODEL = joblib.load(MODEL_PATH)


# bank decision thresholds
APPROVE_PD = 0.60  
REVIEW_PD = 0.75  

def risk_grade(pd):
    if pd < 0.25:
        return "A+ (Prime)"
    elif pd < 0.45:
        return "A (Low Risk)"
    elif pd < 0.60:
        return "B (Acceptable)"
    elif pd < 0.75:
        return "C (Manual Review)"
    else:
        return "D (Reject)"


def loan_decision(pd):
    if pd < APPROVE_PD:
        return "APPROVE"
    elif pd < REVIEW_PD:
        return "REVIEW"
    else:
        return "REJECT"


# django integration
def underwrite_from_django(df):
    """Takes a Pandas DataFrame directly from Django's database."""
    
    df_company = df.copy()
    
    # rename columns if needed
    df_company = df_company.rename(columns={"Date": "date", "Balance": "balance"})
    
    # create description column
    txn_col = "txn_type" if "txn_type" in df_company.columns else "type"
    dest_col = "nameDest" if "nameDest" in df_company.columns else "namedest"
    df_company["description"] = df_company[txn_col].astype(str) + " " + df_company[dest_col].astype(str)
    
    # set correct sign for cash flow
    CREDIT_TYPES = ["CASH_IN"]
    df_company["amount"] = np.where(
        df_company[txn_col].isin(CREDIT_TYPES),
        abs(df_company["amount"]),  
        -abs(df_company["amount"]) 
    )

    df_company = df_company[["date", "description", "amount", "balance"]]

    # generate behavioral features
    tx = load_transactions(df_company)
    bf = behavioural_features(tx)
    
    # match model feature columns
    model_cols = joblib.load(FEATURES_PATH)
    for col in model_cols:
        if col not in bf.columns:
            bf[col] = 0
    bf = bf[model_cols]
    
    # predict and generate explanation
    pd_value = MODEL.predict_proba(bf)[0, 1]
    reasons = generate_explanation(bf.iloc[0])

    return {
        "Credit_Score": pd_to_score(pd_value),
        "Probability_of_Default": round(float(pd_value), 4),
        "Risk_Grade": risk_grade(pd_value),
        "Decision": loan_decision(pd_value),
        "Key_Risk_Drivers": reasons,
        "ml_features": bf.iloc[0].to_dict()
    }


# standalone underwriting
def underwrite(csv_file):

    tx = load_transactions(csv_file)
    bf = behavioural_features(tx)

    pd_value = MODEL.predict_proba(bf)[0,1]

    reasons = generate_explanation(bf.iloc[0])

    result = {
        "Credit_Score": pd_to_score(pd_value),
        "Probability_of_Default": round(float(pd_value),4),
        "Risk_Grade": risk_grade(pd_value),
        "Decision": loan_decision(pd_value),
        "Key_Risk_Drivers": reasons
    }

    return result


def pd_to_score(pd):
    score = 850 - (pd * 550)
    return int(max(300, min(850, score)))


import sys


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python underwrite.py <transaction_csv>")
    else:
        file_path = sys.argv[1]
        result = underwrite(file_path)
        print("\n===== UNDERWRITING RESULT =====\n")
        for k, v in result.items():
            if isinstance(v, list):
                print(f"{k}:")
                for r in v:
                    print("  -", r)
            else:
                print(f"{k}: {v}")