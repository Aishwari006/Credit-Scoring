import pandas as pd


df_raw = pd.read_csv("dataset/paysim_dataset.csv",
                 nrows=1000000)

import pandas as pd
import numpy as np
from datetime import timedelta


print(" Generating Multi-Tenant MSME Database...")


# assume 1M PaySim dataset already loaded
company_ids = [f"MSME_{i:03d}" for i in range(1,501)]
all_companies_data = []


# base start date
base_date = pd.Timestamp("2021-01-01 08:00:00")


profiles = {
    "strong_stable": 0.30,
    "healthy": 0.25,
    "seasonal": 0.20,
    "volatile": 0.15,
    "distressed": 0.10
}


profile_list = np.random.choice(
    list(profiles.keys()),
    size=len(company_ids),
    p=list(profiles.values())
)


for company, profile in zip(company_ids, profile_list):

    txn_count = np.random.randint(300, 600)
    df_comp = df_raw.sample(n=txn_count).copy()

    # company size multiplier
    size_multiplier = np.random.choice([0.3, 0.6, 1, 2.5, 5], p=[0.2,0.25,0.3,0.2,0.05])
    df_comp["amount"] *= size_multiplier

    df_comp['nameOrig'] = company
    df_comp = df_comp.sort_values(by='step').reset_index(drop=True)

    # long transaction history
    years = np.random.randint(3, 6)
    days_total = years * 365

    # distribute transactions across years
    day_positions = np.sort(np.random.uniform(0, days_total, txn_count))
    df_comp['Date'] = base_date + pd.to_timedelta(day_positions, unit="D")

    # strong stable companies
    if profile == "strong_stable":

        # increase revenue
        cashin_idx = df_comp["type"] == "CASH_IN"
        df_comp.loc[cashin_idx, "amount"] *= np.random.uniform(1.4, 2.2)

        # reduce expenses
        expense_idx = df_comp["type"] != "CASH_IN"
        df_comp.loc[expense_idx, "amount"] *= np.random.uniform(0.3, 0.6)

        # diversified customers
        df_comp["nameDest"] = "MULTI_CUSTOMERS_" + str(np.random.randint(1,50))

        pass

    # profile behavior adjustments
    if profile == "growing":
        df_comp["amount"] *= np.linspace(0.8, 1.8, len(df_comp))

    elif profile == "healthy":

        # steady growth
        df_comp["amount"] *= np.linspace(1.0, 1.4, len(df_comp))

        # small random shocks
        shock_idx = df_comp.sample(frac=0.05).index
        df_comp.loc[shock_idx, "amount"] *= 1.5

    elif profile == "seasonal":
        seasonal = 1 + 0.5*np.sin(np.linspace(0, 8*np.pi, len(df_comp)))
        df_comp["amount"] *= seasonal

    elif profile == "volatile":
        shock_idx = df_comp.sample(frac=0.25).index
        df_comp.loc[shock_idx, "amount"] *= np.random.uniform(2,5)

    elif profile == "distressed":

        # decreasing revenue
        cashin_idx = df_comp["type"] == "CASH_IN"
        n_in = cashin_idx.sum()
        if n_in > 0:
            df_comp.loc[cashin_idx, "amount"] *= np.linspace(1, 0.2, n_in)

        # increasing expenses
        expense_idx = df_comp["type"] != "CASH_IN"
        n_out = expense_idx.sum()
        if n_out > 0:
            df_comp.loc[expense_idx, "amount"] *= np.linspace(1, 2.5, n_out)

        # single vendor dependency
        df_comp["nameDest"] = "MAIN_VENDOR"

        # large random shocks
        crash_idx = df_comp.sample(frac=0.2).index
        df_comp.loc[crash_idx, "amount"] *= 4

        # income freeze period
        freeze_start = int(len(df_comp)*0.55)
        freeze_end = int(len(df_comp)*0.75)

        mask = (df_comp.index >= freeze_start) & (df_comp.index <= freeze_end) & (df_comp["type"]=="CASH_IN")
        df_comp.loc[mask, "amount"] *= 0.05

    # recompute balances
    starting_balance = np.random.uniform(50000, 200000)
    balances = []
    current_balance = starting_balance

    for index, row in df_comp.iterrows():
        amount = row['amount']
        txn_type = row['type']

        # accounting logic
        if txn_type == 'CASH_IN':
            current_balance += amount

        elif txn_type in ['PAYMENT', 'CASH_OUT', 'DEBIT']:

            if profile != "distressed":
                # protect liquidity
                if current_balance - amount < 500:
                    amount = current_balance * np.random.uniform(0.2, 0.6)
                    df_comp.at[index, 'amount'] = amount

            # distressed companies overspend
            current_balance -= amount

        elif txn_type == 'TRANSFER':
            # no balance impact
            pass

        # prevent extreme negative spiral
        if profile == "distressed":
            current_balance = max(current_balance, -50000)
        else:
            current_balance = max(current_balance, 100)

        balances.append(current_balance)

    df_comp['Balance'] = balances
    all_companies_data.append(df_comp)


# combine all companies
df_master_db = pd.concat(all_companies_data, ignore_index=True)

# keep only required columns
cols_to_keep = ['nameOrig', 'Date', 'type', 'amount', 'nameDest', 'Balance']
df_master_db = df_master_db[cols_to_keep]

print(f" Created Master Database with {len(df_master_db)} total transactions across {len(company_ids)} companies.")
print(df_master_db.head())
print(df_master_db['nameOrig'].value_counts())

df_master_db["ground_truth_profile"] = df_master_db["nameOrig"].map(
    dict(zip(company_ids, profile_list))
)

# save master dataset
df_master_db.to_csv("Multi_Company_Raw_Transactions.csv", index=False)

df_master_db.groupby('nameOrig')['Balance'].std().describe()