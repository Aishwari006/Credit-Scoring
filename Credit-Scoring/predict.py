import os
import pandas as pd
import numpy as np
import joblib


from mapping_layer import process_company


# =========================================================
# PATH SETUP (DJANGO SAFE)
# =========================================================
# This ensures Django finds your .pkl files no matter where the server is launched
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")


# =========================================================
# LOAD TRAINED ARTIFACTS
# =========================================================
imputer = joblib.load(os.path.join(MODEL_DIR, "01_imputer.pkl"))
scaler = joblib.load(os.path.join(MODEL_DIR, "02_scaler.pkl"))
model = joblib.load(os.path.join(MODEL_DIR, "03_scorecard_model.pkl"))
threshold = float(joblib.load(os.path.join(MODEL_DIR, "06_decision_threshold.pkl")))
score_scaling = joblib.load(os.path.join(MODEL_DIR, "07_scorecard_scaling.pkl"))
model_columns = joblib.load(os.path.join(MODEL_DIR, "05_model_input_columns.pkl"))


# Make sure you added joblib.dump(coefficients, "09_model_coefficients.pkl") to train.py!
coefficients = joblib.load(os.path.join(MODEL_DIR, "09_model_coefficients.pkl"))


coefficients = pd.Series(coefficients, index=model_columns)


FACTOR = score_scaling["factor"]
OFFSET = score_scaling["offset"]


# =========================================================
# CREDIT SCORE CONVERSION
# =========================================================
def prob_to_score(pd_value):
    pd_value = np.clip(pd_value, 1e-6, 1-1e-6)
    odds = pd_value / (1 - pd_value)
    score = OFFSET - FACTOR * np.log(odds)
    return float(np.clip(score, 300, 900))


# =========================================================
# RISK BUCKET LOGIC
# =========================================================
def risk_bucket(pd_value):
    if pd_value < threshold * 0.5:
        return "Very Low Risk"
    elif pd_value < threshold:
        return "Low Risk (Approved)"
    elif pd_value < threshold * 1.6:
        return "Medium Risk (Borderline)"
    else:
        return "High Risk (Reject)"


# =========================================================
# TRUE FEATURE CONTRIBUTION EXPLANATION
# =========================================================
# This translates the Chinese MSME tabular columns back into the
# behavioral realities that generated them via your mapping_layer.
feature_reason_map = {
    "Registered_capital (Ten thousand Yuan)": "Low operating balance relative to recurring obligations",
    "Paid_in_capital (Ten thousand Yuan)": "Low overall liquidity reserves",
    "t-1 Basic old-age insurance for urban employees": "Unstable monthly revenue pattern",
    "t-1 Basic medical insurance for employees": "Drop in cash flow consistency",
    "Ratepaying_Credit_Grade_A_num": "Sharp balance drops and financial stress detected",
    "Legal_proceedings_num_1year": "Frequent unusually large expense events",
    "Branch_num": "Irregular or low transaction volume",
    "MS_num": "Irregular transaction activity",
    "SH_num": "Narrow customer and vendor network",
    "CL_3years": "Highly unstable monthly credit trend"
}


def extract_risk_drivers(X_scaled, top_n=3):
    """
    Identifies which features pushed probability of default higher.
    contribution = coefficient * scaled_feature_value
    """


    row = X_scaled.iloc[0]
    contributions = row * coefficients


    # Only features increasing default probability
    risk_contrib = contributions[contributions > 0]


    if risk_contrib.empty:
        return ["Financial behaviour appears highly stable"]


    top = risk_contrib.sort_values(ascending=False).head(top_n)


    reasons = [
        feature_reason_map.get(feature, feature.replace('_', ' '))
        for feature in top.index
    ]


    return reasons




# =========================================================
# MAIN PREDICTION PIPELINE
# =========================================================
def predict_company(master_csv, company_id):
    # Step 1 — generate model features from raw transactions
    model_df = process_company(master_csv, company_id)


    # Ensure column order
    model_df = model_df[model_columns]


    # Step 2 — preprocessing (same as training)
    # model_df = model_df.fillna(0)
    X_imp = pd.DataFrame(imputer.transform(model_df), columns=model_columns)
    X_scaled = pd.DataFrame(scaler.transform(X_imp), columns=model_columns)


    # Step 3 — prediction
    pd_value = float(model.predict_proba(X_scaled)[0, 1])
    decision = int(pd_value >= threshold)


    # Step 4 — score mapping
    score = prob_to_score(pd_value)
    bucket = risk_bucket(pd_value)


    # Step 5 — explanation
    reasons = extract_risk_drivers(X_scaled)


    # Step 6 — final structured output
    result = {
        "company_id": company_id,
        "probability_of_default": round(pd_value, 4),
        "credit_score": int(round(score)),
        "decision_threshold": float(threshold),
        "approval": "Rejected" if decision else "Approved",
        "risk_bucket": bucket,
        "top_risk_drivers": reasons
    }


    return result

# =========================================================
# DJANGO INTEGRATION PIPELINE
# =========================================================
def predict_from_django(ml_features_dict):
    """
    Takes the 25 mapped features directly from Django's database,
    bypassing the need for CSV files entirely.
    """
    # 1. Convert dict to DataFrame
    model_df = pd.DataFrame([ml_features_dict])[model_columns]

    # 2. Preprocessing
    X_imp = pd.DataFrame(imputer.transform(model_df), columns=model_columns)
    X_scaled = pd.DataFrame(scaler.transform(X_imp), columns=model_columns)

    # 3. Prediction
    pd_value = float(model.predict_proba(X_scaled)[0, 1])
    decision = int(pd_value >= threshold)

    # 4. Score & Bucket
    score = prob_to_score(pd_value)
    bucket = risk_bucket(pd_value)

    # 5. Explanation
    reasons = extract_risk_drivers(X_scaled)

    # 6. Return Clean Dictionary to Django
    return {
        "probability_of_default": round(pd_value, 4),
        "credit_score": int(round(score)),
        "decision_threshold": float(threshold),
        "approval": "Rejected" if decision else "Approved",
        "risk_bucket": bucket,
        "top_risk_drivers": reasons
    }
    
# =========================================================
# LOCAL TESTING ENTRYPOINT
# =========================================================
if __name__ == "__main__":
    test_csv = os.path.join(BASE_DIR, "Multi_Company_Raw_Transactions.csv")


    companies = [
        "MSME_RETAIL_001",
        "MSME_TECH_002",
        "MSME_AGRI_003",
        "MSME_MANUF_004",
        "MSME_SERVICES_005"
    ]


    if os.path.exists(test_csv):
        print("\n=========== MULTI-COMPANY TEST ===========\n")


        for company in companies:
            print(f"\n---- {company} ----")
            output = predict_company(test_csv, company)


            for k, v in output.items():
                if isinstance(v, list):
                    print(f"{k}:")
                    for r in v:
                        print(f"  - {r}")
                else:
                    print(f"{k}: {v}")
    else:
        print(f"Test file not found: {test_csv}")