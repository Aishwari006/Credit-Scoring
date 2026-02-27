import pandas as pd


df_raw = pd.read_csv("dataset/paysim_dataset.csv",
                 nrows=1000000)

import pandas as pd
import numpy as np
from datetime import timedelta


print("⏳ Generating Multi-Tenant MSME Database...")


# Assuming your raw 1M row PaySim dataframe is loaded as `df_raw`
company_ids = [f"MSME_{i:03d}" for i in range(1,501)]
all_companies_data = []


# Base start date for our simulations
base_date = pd.Timestamp("2021-01-01 08:00:00")


profiles = {
    "strong_stable": 0.30,     # never defaults
    "healthy": 0.25,           # rarely defaults
    "seasonal": 0.20,          # sometimes defaults
    "volatile": 0.15,          # medium risk
    "distressed": 0.10         # high risk
}


profile_list = np.random.choice(
    list(profiles.keys()),
    size=len(company_ids),
    p=list(profiles.values())
)


for company, profile in zip(company_ids, profile_list):


    txn_count = np.random.randint(300, 600)
    df_comp = df_raw.sample(n=txn_count).copy()


    # company scale (micro vs medium enterprise)
    size_multiplier = np.random.choice([0.3, 0.6, 1, 2.5, 5], p=[0.2,0.25,0.3,0.2,0.05])
    df_comp["amount"] *= size_multiplier


    df_comp['nameOrig'] = company
    df_comp = df_comp.sort_values(by='step').reset_index(drop=True)


    # realistic time spacing
    # -------- LONG CREDIT HISTORY (CRITICAL FIX) --------
    years = np.random.randint(3, 6)  # 3–5 years history like real bureau data
    days_total = years * 365


    # distribute transactions across years realistically
    day_positions = np.sort(np.random.uniform(0, days_total, txn_count))
    df_comp['Date'] = base_date + pd.to_timedelta(day_positions, unit="D")


    # ------------------------------------------------
    # STRONG BUSINESSES (guaranteed survivors)
    # ------------------------------------------------
    if profile == "strong_stable":


        # consistent revenue
        cashin_idx = df_comp["type"] == "CASH_IN"
        df_comp.loc[cashin_idx, "amount"] *= np.random.uniform(1.4, 2.2)


        # low expenses
        expense_idx = df_comp["type"] != "CASH_IN"
        df_comp.loc[expense_idx, "amount"] *= np.random.uniform(0.3, 0.6)


        # wide customer base
        df_comp["nameDest"] = "MULTI_CUSTOMERS_" + str(np.random.randint(1,50))


        # remove shocks
        pass
   
    # ---------------- PROFILE BEHAVIOUR ----------------
    if profile == "growing":
        df_comp["amount"] *= np.linspace(0.8, 1.8, len(df_comp))
   
    elif profile == "healthy":


        # steady positive growth
        df_comp["amount"] *= np.linspace(1.0, 1.4, len(df_comp))


        # occasional expenses but safe
        shock_idx = df_comp.sample(frac=0.05).index
        df_comp.loc[shock_idx, "amount"] *= 1.5


    elif profile == "seasonal":
        seasonal = 1 + 0.5*np.sin(np.linspace(0, 8*np.pi, len(df_comp)))
        df_comp["amount"] *= seasonal


    elif profile == "volatile":
        shock_idx = df_comp.sample(frac=0.25).index
        df_comp.loc[shock_idx, "amount"] *= np.random.uniform(2,5)


    elif profile == "distressed":


        # --- collapsing revenue ---
        cashin_idx = df_comp["type"] == "CASH_IN"
        n_in = cashin_idx.sum()
        if n_in > 0:
            df_comp.loc[cashin_idx, "amount"] *= np.linspace(1, 0.2, n_in)


        # --- rising expenses ---
        expense_idx = df_comp["type"] != "CASH_IN"
        n_out = expense_idx.sum()
        if n_out > 0:
            df_comp.loc[expense_idx, "amount"] *= np.linspace(1, 2.5, n_out)


        # vendor dependency (single buyer risk)
        df_comp["nameDest"] = "MAIN_VENDOR"


        # financial shocks (sudden large payments)
        crash_idx = df_comp.sample(frac=0.2).index
        df_comp.loc[crash_idx, "amount"] *= 4


        # liquidity starvation period (no income for a while)
        freeze_start = int(len(df_comp)*0.55)
        freeze_end = int(len(df_comp)*0.75)


        mask = (df_comp.index >= freeze_start) & (df_comp.index <= freeze_end) & (df_comp["type"]=="CASH_IN")
        df_comp.loc[mask, "amount"] *= 0.05


    # ---------------- RECOMPUTE BALANCES ----------------
    starting_balance = np.random.uniform(50000, 200000)
    balances = []
    current_balance = starting_balance


    for index, row in df_comp.iterrows():
        amount = row['amount']
        txn_type = row['type']


        # ---------------- ACCOUNTING LOGIC ----------------
        if txn_type == 'CASH_IN':
            current_balance += amount


        elif txn_type in ['PAYMENT', 'CASH_OUT', 'DEBIT']:


            if profile != "distressed":
                # normal businesses protect liquidity
                if current_balance - amount < 500:
                    amount = current_balance * np.random.uniform(0.2, 0.6)
                    df_comp.at[index, 'amount'] = amount


            # distressed companies DO overspend (real default behaviour)
            current_balance -= amount


        elif txn_type == 'TRANSFER':
            # internal movement → no liquidity impact
            pass


        # prevent negative bankruptcy spiral
        # allow stress but prevent infinite negative spiral
        if profile == "distressed":
            current_balance = max(current_balance, -50000)  # overdraft allowed
        else:
            current_balance = max(current_balance, 100)


        balances.append(current_balance)


    df_comp['Balance'] = balances
    all_companies_data.append(df_comp)


# Combine all companies into a single Master Database DataFrame
df_master_db = pd.concat(all_companies_data, ignore_index=True)


# Clean up columns for the final DB1 format
cols_to_keep = ['nameOrig', 'Date', 'type', 'amount', 'nameDest', 'Balance']
df_master_db = df_master_db[cols_to_keep]


print(f"✅ Created Master Database with {len(df_master_db)} total transactions across {len(company_ids)} companies.")
print(df_master_db.head())
print(df_master_db['nameOrig'].value_counts())


df_master_db["ground_truth_profile"] = df_master_db["nameOrig"].map(
    dict(zip(company_ids, profile_list))
)


# Save this so you can use it to test your Django Uploads
df_master_db.to_csv("Multi_Company_Raw_Transactions.csv", index=False)


df_master_db.groupby('nameOrig')['Balance'].std().describe()


