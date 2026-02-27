import pandas as pd
import numpy as np
import re
import sys
from pathlib import Path


EPS = 1e-6




# =========================================================
# CLEAN DESCRIPTION
# =========================================================
def clean_description(text):
    text = str(text).lower()
    text = re.sub(r'\b(upi|neft|imps|rtgs|txn|ref|transfer)\b', ' ', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^a-z ]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text




# =========================================================
# LOAD TRANSACTIONS
# =========================================================
# COPY THIS INTO behaviour_engine.py REPLACING THE OLD load_transactions FUNCTION


def load_transactions(input_data):
    # Handle both file path (string) and DataFrame inputs
    if isinstance(input_data, str):
        df = pd.read_csv(input_data)
    elif isinstance(input_data, pd.DataFrame):
        df = input_data.copy()
    else:
        raise ValueError("Input must be a CSV file path or pandas DataFrame")


    df.columns = [c.lower().strip() for c in df.columns]


    required = ["date", "description", "amount", "balance"]
    for col in required:
        if col not in df.columns:
            raise Exception(f"Missing required column: {col}")


    # Robust date parsing
    df["date"] = pd.to_datetime(df["date"], errors="coerce")


    # Try Excel serial fallback
    if df["date"].isna().mean() > 0.3:
        try:
            df["date"] = pd.to_datetime("1899-12-30") + pd.to_timedelta(df["date"].astype(float), unit="D")
        except:
            pass


    if df["date"].isna().sum() > 0:
        raise Exception("Could not parse date column — unsupported format")


    df = df.sort_values("date")
    # SAFETY: detect unsigned transaction datasets
   
   
    # -------------------------------------------------
    # AUTO-SIGN AMOUNTS FOR REAL BANK STATEMENTS
    # -------------------------------------------------
    # If dataset has no negative values, infer debit vs credit
    if (df["amount"] < 0).sum() == 0:


        print("Unsigned transactions detected → inferring debit/credit...")


        debit_keywords = [
            "payment","rent","emi","loan","transfer","sent",
            "debit","withdraw","atm","upi-","neft-","imps-",
            "charge","fee","gst","tax","bill","purchase",
            "electricity","water","fuel","salary","vendor"
        ]


        credit_keywords = [
            "received","deposit","sale","customer","credit","refund"
        ]


        def infer_sign(row):
            text = str(row["description"]).lower()


            if any(k in text for k in debit_keywords):
                return -abs(row["amount"])


            if any(k in text for k in credit_keywords):
                return abs(row["amount"])


            # fallback: use balance movement
            return row["amount"]


        df["amount"] = df.apply(infer_sign, axis=1)


        # FINAL CHECK (after fixing)
        if (df["amount"] < 0).sum() == 0:
            raise Exception(
                "Could not infer debit transactions from descriptions. "
                "Your demo_company.csv descriptions are unrealistic."
            )


    df["credit"] = df["amount"].clip(lower=0)
    df["debit"] = -df["amount"].clip(upper=0)


    return df


# =========================================================
# FEATURE ENGINE
# =========================================================
def behavioural_features(df):


    g = df.copy()
    features = {}


    history_days = (g["date"].max() - g["date"].min()).days + 1
    features["history_length_days"] = history_days


   


    # ---------------- DAILY BALANCE CURVE ----------------
    g = g.sort_values("date")


    start = g["date"].min().normalize()
    end = g["date"].max().normalize()


    calendar = pd.date_range(start, end, freq="D")


    daily_balance = (
        g.set_index("date")["balance"]
        .resample("D")
        .last()
        .reindex(calendar)
        .ffill()
    )


    # NEW — prevent missing-history bias
    if daily_balance.isna().all():
        daily_balance[:] = g["balance"].iloc[0]
    elif daily_balance.isna().sum() > 0:
        first_valid = daily_balance.dropna().iloc[0]
        daily_balance = daily_balance.fillna(first_valid)


    observed_days = (g["date"].max() - g["date"].min()).days + 1
    months = max(observed_days / 30.44, 1)


    # features = {}


    # ---------------- LIQUIDITY ----------------
    features["avg_balance"] = daily_balance.mean()
    features["min_balance"] = daily_balance.min()
    features["max_balance"] = daily_balance.max()


    features["balance_volatility"] = daily_balance.std()
    features["balance_cv"] = (
        features["balance_volatility"] /
        (abs(features["avg_balance"]) + EPS)
    )


    features["negative_balance_days"] = (daily_balance < 0).sum()
    liquidity_floor = daily_balance.quantile(0.2)
    features["low_balance_days"] = (daily_balance < liquidity_floor).sum()
    # ---------------- ACTIVITY ----------------
    features["txn_per_month"] = len(g) / months
    observed_days = (g["date"].max() - g["date"].min()).days + 1
    features["active_days_ratio"] = g["date"].dt.date.nunique() / max(observed_days,1)


    # ---------------- MONTHLY CREDIT & DEBIT ----------------
    monthly_credit = g.groupby(g["date"].dt.to_period("M"))["credit"].sum()
    monthly_debit = g.groupby(g["date"].dt.to_period("M"))["debit"].sum()


    # Identify if the first or last calendar months are cut off
    min_date = g["date"].min()
    max_date = g["date"].max()


    # Drop the first month if the statement started after the 5th
    if min_date.day > 5:
        monthly_credit = monthly_credit.drop(min_date.to_period("M"), errors="ignore")
        monthly_debit = monthly_debit.drop(min_date.to_period("M"), errors="ignore")


    # Drop the last month if the statement ended before the 25th
    if max_date.day < 25:
        monthly_credit = monthly_credit.drop(max_date.to_period("M"), errors="ignore")
        monthly_debit = monthly_debit.drop(max_date.to_period("M"), errors="ignore")


    # Fallbacks for empty data
    if len(monthly_credit) == 0:
        monthly_credit = pd.Series([0])
    if len(monthly_debit) == 0:
        monthly_debit = pd.Series([0])


    # Credit Features
    features["avg_monthly_credit"] = monthly_credit.mean()
    features["credit_std"] = monthly_credit.std() if len(monthly_credit) > 1 else 0
    features["credit_stability"] = features["avg_monthly_credit"] / (features["credit_std"] + EPS)


    if len(monthly_credit) > 1:
        first = monthly_credit.iloc[0] + EPS
        last = monthly_credit.iloc[-1] + EPS
        features["credit_trend"] = np.log(last / first)
    else:
        features["credit_trend"] = 0


    # Debit Features
    features["avg_monthly_debit"] = monthly_debit.mean()
    features["debit_credit_ratio"] = features["avg_monthly_debit"] / (features["avg_monthly_credit"] + EPS)


    # ---------------- RECURRING OBLIGATIONS ----------------
    # ---------------- RECURRING OBLIGATIONS ----------------
    def detect_recurring(g):
        debits = g[g["debit"] > 0].copy()
        if debits.empty:
            return 0


        debits["month"] = debits["date"].dt.to_period("M")
        recurring_candidates = []
        visited_idx = set()


        # Sort descending so we cluster around the largest anchor amounts first
        for amount in sorted(debits["debit"].unique(), reverse=True):
           
            # Mask: within 5% AND not already part of another recurring cluster
            similar_mask = (np.abs(debits["debit"] - amount) / (amount + EPS) < 0.05) & (~debits.index.isin(visited_idx))
            similar = debits[similar_mask]


            day_std = similar["date"].dt.day.std()
            day_consistency = (day_std < 4) if not np.isnan(day_std) else False


            if similar["month"].nunique() >= 3 and day_consistency:
                recurring_candidates.append(similar.groupby("month")["debit"].mean().mean())
                # Mark these specific transactions as processed
                visited_idx.update(similar.index.tolist())


        return np.sum(recurring_candidates)
   
    features["avg_recurring_monthly"] = detect_recurring(g)
    features["fixed_obligation_ratio"] = features["avg_recurring_monthly"] / (features["avg_monthly_credit"] + EPS)


   
    # ---------------- SHOCK EVENTS ----------------
    if len(g[g["debit"] > 0]) < 20:
        q = 0.85
    else:
        q = 0.97


    large_debit = g["debit"] > g["debit"].quantile(q)
    features["large_debit_events"] = large_debit.sum()


    window = min(7, max(3, len(daily_balance)//4))
    rolling_mean = daily_balance.rolling(window, min_periods=1).mean()


    threshold = 0.7 if len(daily_balance) > 30 else 0.55
    balance_drop = daily_balance < threshold * rolling_mean


    features["sudden_drop_events"] = ((balance_drop) & (~balance_drop.shift(1).fillna(False))).sum()


    # ---------------- COUNTERPARTIES ----------------
    g["clean_desc"] = g["description"].apply(clean_description)
    features["unique_counterparties"] = g["clean_desc"].nunique()


    # ---------------- INACTIVITY ----------------
    gaps = g["date"].diff().dt.days.fillna(0)
    features["max_inactive_gap"] = gaps.max()


    # =========================================================
    # >>> ADD THIS WHOLE BLOCK RIGHT HERE <<<
    # FINANCIAL SURVIVAL SIGNALS (USED BY RISK MODEL)
    # =========================================================


    credits = g[g["amount"] > 0]["amount"]
    debits = g[g["amount"] < 0]["amount"].abs()


    features["avg_credit"] = credits.mean() if len(credits) else 0
    features["avg_debit"] = debits.mean() if len(debits) else 0
    features["max_debit"] = debits.max() if len(debits) else 0


    features["balance_std"] = daily_balance.std()


    # months with no income
    monthly_credit_full = g.groupby(g["date"].dt.to_period("M"))["credit"].sum()
    features["months_without_credit"] = (monthly_credit_full == 0).sum()


    # dependency on single counterparty
    counterparty_share = g["clean_desc"].value_counts(normalize=True)
    features["top_counterparty_share"] = counterparty_share.iloc[0] if len(counterparty_share) else 0


    return pd.DataFrame([features])




# =========================================================
# MAIN EXECUTION
# =========================================================
def main():


    if len(sys.argv) < 2:
        print("Usage: python behaviour_engine.py <transactions.csv>")
        return


    input_path = sys.argv[1]
    df = load_transactions(input_path)


    features = behavioural_features(df)


    output_path = Path("outputs/customer_features.csv")
    output_path.parent.mkdir(exist_ok=True)


    features.to_csv(output_path, index=False)


    print("\nBehaviour profiling complete")
    print("Saved:", output_path)




if __name__ == "__main__":
    main()

