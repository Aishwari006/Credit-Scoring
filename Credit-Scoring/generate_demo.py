import pandas as pd
import numpy as np
from datetime import timedelta

np.random.seed(42)

def generate_company(filename, is_risky=False):
    # generate transactions every 2 days for 2 years
    dates = pd.date_range(start="2022-01-01", end="2023-12-31", freq="2D")
    
    balance = 100000 if not is_risky else 60000
    data = []
    
    data.append(["2022-01-01", "Opening Balance", balance, balance])
    
    for i, date in enumerate(dates[1:]):
        month_progress = i / len(dates)  # progress over time
        
        if i % 3 == 0:
            # income transactions
            desc = np.random.choice(["UPI RECEIVED CUSTOMER", "AMAZON SALE CREDIT", "SWIGGY SETTLEMENT CREDIT"])
            if is_risky:
                # revenue decreases over time
                amount = int(np.random.uniform(15000, 30000) * (1 - month_progress * 0.7))
            else:
                # revenue increases over time
                amount = int(np.random.uniform(30000, 50000) * (1 + month_progress * 0.4))
            balance += amount
        else:
            # expense transactions
            desc = np.random.choice(["SHOP RENT PAYMENT", "SALARY STAFF PAYMENT", "VENDOR PURCHASE PAYMENT", "ELECTRICITY BILL PAYMENT", "GST TAX PAYMENT"])
            if is_risky:
                # expenses increase and penalties may occur
                amount = int(np.random.uniform(20000, 35000) * (1 + month_progress * 0.5))
                if np.random.random() < 0.15:  # 15% penalty chance
                    desc = "OVERDRAFT PENALTY FEE"
                    amount += 15000
            else:
                # stable expenses
                amount = int(np.random.uniform(10000, 20000))
                
            balance -= amount
            
        data.append([date.strftime("%Y-%m-%d"), desc, amount, balance])
        
    df = pd.DataFrame(data, columns=["date", "description", "amount", "balance"])
    df.to_csv(filename, index=False)
    print(f" Generated {filename} | Rows: {len(df)} | Final Balance: ₹{balance:,.0f}")

# run generator
generate_company("healthy_company.csv", is_risky=False)
generate_company("risky_company.csv", is_risky=True)