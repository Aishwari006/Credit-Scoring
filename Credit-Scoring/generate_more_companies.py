import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

def generate_bank_statement(filename, start_balance, profile_type):
    # generate 24 months of transactions
    start_date = datetime(2024, 1, 1)
    end_date = datetime(2025, 12, 31)
    
    current_date = start_date
    current_balance = start_balance
    transactions = []
    
    # description keywords used for inflow and outflow detection
    
    while current_date <= end_date:
        month_progress = (current_date.year - 2024) * 12 + current_date.month
        
        # steady service business
        if profile_type == 'steady_service':
            inflow_chance, outflow_chance = 0.2, 0.3
            inflow_amt = random.uniform(2000, 3000)
            outflow_amt = random.uniform(500, 1000)
            desc_in = random.choice(["Monthly Retainer Customer", "Consulting Sale Received"])
            desc_out = random.choice(["Office Rent", "Software Vendor", "Internet Utility Bill"])
            
        # hyper growth startup
        elif profile_type == 'hyper_growth':
            inflow_chance, outflow_chance = 0.1, 0.6
            inflow_amt = random.uniform(5000, 8000) 
            
            # occasional funding deposit
            if random.random() < 0.02:
                inflow_amt = 150000 
                desc_in = "Seed Funding Deposit"
            else:
                desc_in = "Software Subscription Sale"
                
            outflow_amt = random.uniform(3000, 7000)
            desc_out = random.choice(["Cloud Server Vendor", "Engineering Payroll", "Marketing Agency Vendor"])

        # declining retail business
        elif profile_type == 'declining_retail':
            inflow_chance, outflow_chance = 0.4, 0.4
            
            # sales decrease over time
            sales_multiplier = max(0.1, 1.0 - (month_progress * 0.04)) 
            inflow_amt = random.uniform(1000, 3000) * sales_multiplier
            outflow_amt = random.uniform(1500, 2500)
            desc_in = random.choice(["POS Daily Sale", "Customer Payment Received"])
            desc_out = random.choice(["Retail Space Rent", "Staff Payroll", "Inventory Purchase Vendor"])

        # generate inflow
        if random.random() < inflow_chance:
            current_balance += inflow_amt
            transactions.append({
                "date": current_date.strftime("%Y-%m-%d"),
                "description": desc_in,
                "amount": round(inflow_amt, 2),
                "balance": round(current_balance, 2)
            })
            
        # generate outflow
        if random.random() < outflow_chance:
            current_balance -= outflow_amt
            transactions.append({
                "date": current_date.strftime("%Y-%m-%d"),
                "description": desc_out,
                "amount": round(outflow_amt, 2),
                "balance": round(current_balance, 2)
            })
            
        current_date += timedelta(days=1)

    # save to csv
    df = pd.DataFrame(transactions)
    
    # match expected column order
    df = df[['date', 'description', 'amount', 'balance']]
    df.to_csv(filename, index=False)
    
    print(f"Generated {filename} with {len(df)} transactions. Final Balance: ₹{current_balance:,.2f}")

# run generators
if __name__ == "__main__":
    generate_bank_statement("steady_service.csv", start_balance=15000, profile_type="steady_service")
    generate_bank_statement("hypergrowth_startup.csv", start_balance=200000, profile_type="hyper_growth")
    generate_bank_statement("declining_retail.csv", start_balance=40000, profile_type="declining_retail")