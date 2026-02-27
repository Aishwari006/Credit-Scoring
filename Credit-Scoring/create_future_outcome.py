import pandas as pd
import numpy as np
from behaviour_engine import behavioural_features, load_transactions


OBSERVATION_RATIO = 0.7   # portion used as past data
MIN_HISTORY = 90          # minimum required history in days


# check if company defaulted after observation period
def future_default(transactions):

    if len(transactions) < 30:
        return None

    df = transactions.sort_values("date").reset_index(drop=True)

    split = int(len(df) * OBSERVATION_RATIO)
    past = df.iloc[:split]
    future = df.iloc[split:]

    if len(future) < 10:
        return None

    # calculate failure indicators
    negative_days = (future["balance"] < 0).sum()

    future_credit = future["credit"].sum()
    future_debit = future["debit"].sum()

    if future_credit == 0:
        return 1  # no income, business stopped

    burn_ratio = future_debit / (future_credit + 1e-6)

    severe_loss = future["balance"].min() < 0
    runaway_burn = burn_ratio > 1.5
    persistent_negative = negative_days > len(future)*0.25

    if severe_loss or runaway_burn or persistent_negative:
        return 1
    return 0


# build training dataset from master csv
def build_dataset(master_csv):

    raw = pd.read_csv(master_csv)
    companies = raw["nameOrig"].unique()

    rows = []

    for cid in companies:

        df = raw[raw["nameOrig"] == cid].copy()

        df = df.rename(columns={"Date":"date","Balance":"balance"})
        df["description"] = df["type"] + "_" + df["nameDest"]

        df["amount"] = np.where(df["type"]=="CASH_IN", df["amount"], -df["amount"])

        tx = load_transactions(df)

        label = future_default(tx)
        if label is None:
            continue

        split = int(len(tx)*OBSERVATION_RATIO)
        past_tx = tx.iloc[:split]

        bf = behavioural_features(past_tx).iloc[0]

        row = bf.to_dict()
        row["default"] = label
        row["company"] = cid

        rows.append(row)

    dataset = pd.DataFrame(rows)
    dataset.to_csv("behaviour_training_dataset.csv", index=False)

    print("\nDataset created")
    print("Companies used:", len(dataset))
    print(dataset["default"].value_counts())


if __name__ == "__main__":
    build_dataset("Multi_Company_Raw_Transactions.csv")