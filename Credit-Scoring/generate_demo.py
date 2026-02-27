import pandas as pd
import numpy as np
from datetime import timedelta

np.random.seed(42)

def generate_company(filename, is_risky=False):
    # Generates a transaction every 2-3 days for 2 years
    dates = pd.date_range(start="2022-01-01", end="2023-12-31", freq="2D")
    
    balance = 100000 if not is_risky else 60000
    data = []
    
    data.append(["2022-01-01", "Opening Balance", balance, balance])
    
    for i, date in enumerate(dates[1:]):
        month_progress = i / len(dates) # Progress from 0 to 1 over the 2 years
        
        if i % 3 == 0:
            # --- INCOME LOGIC ---
            desc = np.random.choice(["UPI RECEIVED CUSTOMER", "AMAZON SALE CREDIT", "SWIGGY SETTLEMENT CREDIT"])
            if is_risky:
                # Risky company loses revenue over time
                amount = int(np.random.uniform(15000, 30000) * (1 - month_progress * 0.7))
            else:
                # Healthy company grows revenue
                amount = int(np.random.uniform(30000, 50000) * (1 + month_progress * 0.4))
            balance += amount
        else:
            # --- EXPENSE LOGIC ---
            desc = np.random.choice(["SHOP RENT PAYMENT", "SALARY STAFF PAYMENT", "VENDOR PURCHASE PAYMENT", "ELECTRICITY BILL PAYMENT", "GST TAX PAYMENT"])
            if is_risky:
                # Risky company has rising expenses and shock events
                amount = int(np.random.uniform(20000, 35000) * (1 + month_progress * 0.5))
                if np.random.random() < 0.15: # 15% chance of severe penalty
                    desc = "OVERDRAFT PENALTY FEE"
                    amount += 15000
            else:
                # Healthy company has stable expenses
                amount = int(np.random.uniform(10000, 20000))
                
            balance -= amount
            
        data.append([date.strftime("%Y-%m-%d"), desc, amount, balance])
        
    df = pd.DataFrame(data, columns=["date", "description", "amount", "balance"])
    df.to_csv(filename, index=False)
    print(f"✅ Generated {filename} | Rows: {len(df)} | Final Balance: ₹{balance:,.0f}")

# Run the generator
generate_company("healthy_company.csv", is_risky=False)
generate_company("risky_company.csv", is_risky=True)