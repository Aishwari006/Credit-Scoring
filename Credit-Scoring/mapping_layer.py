import pandas as pd
import numpy as np
import joblib
from behaviour_engine import behavioural_features, load_transactions
import os


EPS = 1e-6
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Force Python to look inside the 'models' subfolder
MODEL_DIR = os.path.join(BASE_DIR, "models")


# ============================================
# Load trained schema
# ============================================
MODEL_COLUMNS = joblib.load(os.path.join(MODEL_DIR, "05_model_input_columns.pkl"))
DEFAULTS = joblib.load(os.path.join(MODEL_DIR, "08_feature_defaults.pkl"))


# ============================================
# Keep mapped features inside realistic ranges
# ============================================
def scale_feature(base, strength, min_factor=0.4, max_factor=1.8):
    factor = 1 + strength
    factor = max(min_factor, min(max_factor, factor))
    return float(base) * factor




# ============================================
# Helper: normalize signal safely (-1 to +1)
# ============================================
def normalize_signal(x, scale=5):
    x = np.tanh(x / scale)
    return float(x)




# ============================================
# Convert behaviour → MSME model features
# ============================================
def behaviour_to_model_features(bf):


    f = {}


    # ---------- CORE RISK SIGNALS ----------
    ratio = (bf["avg_balance"] + EPS) / (bf["avg_recurring_monthly"] + EPS)
    ratio = max(ratio, EPS)
    liquidity_strength = normalize_signal(np.log(ratio))


    stability_strength = normalize_signal(bf["credit_stability"])
    growth_strength = normalize_signal(bf["credit_trend"] * 3)
    stress_strength = normalize_signal(bf["sudden_drop_events"] + bf["large_debit_events"])
    activity_strength = normalize_signal(bf["txn_per_month"] / 50)
    network_strength = normalize_signal(bf["unique_counterparties"] / 40)


    # ---------- CAPITAL ----------
    f['Registered_capital (Ten thousand Yuan)'] = scale_feature(DEFAULTS['Registered_capital (Ten thousand Yuan)'], liquidity_strength)
    f['Paid_in_capital (Ten thousand Yuan)'] = scale_feature(DEFAULTS['Paid_in_capital (Ten thousand Yuan)'], liquidity_strength * 0.7)


    # ---------- EMPLOYEE / PAYROLL ----------
    payroll_factor = (liquidity_strength + stability_strength)/2


    f['t-1 Basic old-age insurance for urban employees'] = scale_feature(DEFAULTS['t-1 Basic old-age insurance for urban employees'], payroll_factor)
    f['t-2 Basic old-age insurance for urban employees'] = scale_feature(DEFAULTS['t-2 Basic old-age insurance for urban employees'], payroll_factor*0.95)
    f['t-3 Basic old-age insurance for urban employees'] = scale_feature(DEFAULTS['t-3 Basic old-age insurance for urban employees'], payroll_factor*0.9)


    f['t-1 Basic medical insurance for employees'] = scale_feature(DEFAULTS['t-1 Basic medical insurance for employees'], payroll_factor*0.85)
    f['t-2 Basic medical insurance for employees'] = scale_feature(DEFAULTS['t-2 Basic medical insurance for employees'], payroll_factor*0.8)


    f['t-1 Unemployment insurance'] = scale_feature(DEFAULTS['t-1 Unemployment insurance'], payroll_factor*0.5)
    f['t-1 Employment injury insurance'] = scale_feature(DEFAULTS['t-1 Employment injury insurance'], payroll_factor*0.45)
    f['t-2 Employment injury insurance'] = scale_feature(DEFAULTS['t-2 Employment injury insurance'], payroll_factor*0.4)
    f['t-1 Birth insurance'] = scale_feature(DEFAULTS['t-1 Birth insurance'], payroll_factor*0.3)
    f['t-2 Birth insurance'] = scale_feature(DEFAULTS['t-2 Birth insurance'], payroll_factor*0.25)


    # ---------- COMPANY SIZE ----------
    f['Branch_num'] = int(max(0,
        scale_feature(DEFAULTS['Branch_num'], activity_strength)
    ))
    f['SH_num'] = int(max(0,
        scale_feature(DEFAULTS['SH_num'], network_strength)
    ))
    f['MS_num'] = int(max(0,
        scale_feature(DEFAULTS['MS_num'], activity_strength)
    ))


    # ---------- CREDIT HISTORY ----------
    f['CL_3years'] = scale_feature(DEFAULTS['CL_3years'], stability_strength)
    f['CL_4years'] = scale_feature(DEFAULTS['CL_4years'], stability_strength*1.1)


    # ---------- TAX QUALITY ----------
    f['Ratepaying_Credit_Grade_A_num'] = int(max(0,
        scale_feature(DEFAULTS['Ratepaying_Credit_Grade_A_num'], -stress_strength)
    ))


    f['Legal_proceedings_num_1year'] = int(max(0,
        scale_feature(DEFAULTS['Legal_proceedings_num_1year'], stress_strength)
    ))


    # ---------- DOCUMENTATION ----------
    f['Certificate_num_2years'] = int(max(0,
        scale_feature(DEFAULTS['Certificate_num_2years'], stability_strength)
    ))
    f['Filing_information_num_2years'] = int(max(0,
        scale_feature(DEFAULTS['Filing_information_num_2years'], stability_strength*0.8)
    ))


    # ---------- INNOVATION ----------
    innovation = (growth_strength + stability_strength)/2


    f['Patent_info_num_2years'] = int(max(0, scale_feature(DEFAULTS['Patent_info_num_2years'] , innovation)))
    f['Patent_info_num_3years'] = int(max(0, scale_feature(DEFAULTS['Patent_info_num_3years'] , innovation*1.1)))
    f['Patent_info_num_5years+'] = int(max(0, scale_feature(DEFAULTS['Patent_info_num_5years+'] , innovation*1.2)))
    f['Trademark_info_num_5years+'] = int(max(0, scale_feature(DEFAULTS['Trademark_info_num_5years+'] , innovation)))


    # ---------- Fill missing safely ----------
    for col in MODEL_COLUMNS:
        if col not in f:
            f[col] = DEFAULTS.get(col, 0)


    return pd.DataFrame([f])[MODEL_COLUMNS]




# ============================================
# MAIN PIPELINE
# ============================================
def process_company(master_csv, company_id):


    df = pd.read_csv(master_csv)
    df_company = df[df["nameOrig"] == company_id].copy()


    if len(df_company) < 30:
        raise Exception("Not enough transaction history")


    df_company = df_company.rename(columns={
        "Date":"date",
        "Balance":"balance"
    })


    # Create description
    df_company["description"] = df_company["type"] + "_" + df_company["nameDest"]


    # Convert PaySim absolute amounts → signed amounts
    # Incoming money = positive
    # Outgoing money = negative
    CREDIT_TYPES = ["CASH_IN"]
    DEBIT_TYPES = ["CASH_OUT", "PAYMENT", "TRANSFER", "DEBIT"]


    df_company["amount"] = np.where(
        df_company["type"].isin(CREDIT_TYPES),
        df_company["amount"],      # incoming
        -df_company["amount"]      # outgoing
    )


    df_company = df_company[["date","description","amount","balance"]]


    tx = load_transactions(df_company)
    bf = behavioural_features(tx).iloc[0]


    model_df = behaviour_to_model_features(bf)
    model_df.to_csv("model_ready_features.csv", index=False)


    print("Model features generated for:", company_id)
    return model_df


# ============================================
# DJANGO BRIDGE (For views.py)
# ============================================
def process_django_dataframe(df_company):
    if len(df_company) < 30:
        raise Exception("Not enough transaction history")

    df_company = df_company.rename(columns={
        "Date": "date",
        "Balance": "balance"
    })

    # Create description
    df_company["description"] = df_company["txn_type"] + "_" + df_company["nameDest"].fillna("Unknown")

    # Convert absolute amounts to signed amounts
    CREDIT_TYPES = ["CASH_IN"]
    df_company["amount"] = np.where(
        df_company["txn_type"].isin(CREDIT_TYPES),
        df_company["amount"],      # incoming
        -df_company["amount"]      # outgoing
    )

    df_company = df_company[["date","description","amount","balance"]]

    tx = load_transactions(df_company)
    bf = behavioural_features(tx).iloc[0]

    model_df = behaviour_to_model_features(bf)
    return model_df.iloc[0].to_dict()

if __name__ == "__main__":
    # A safe test block for your friend to use independently of Django
    try:
        print("Testing mapping layer with local CSV...")
        process_company("Multi_Company_Raw_Transactions.csv", "MSME_RETAIL_001")
        print("Success! Features mapped.")
    except FileNotFoundError:
        print("Notice: Master CSV not found. This script is now driven by the Django frontend database.")
