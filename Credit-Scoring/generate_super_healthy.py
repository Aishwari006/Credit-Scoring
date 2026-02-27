import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

def generate_super_healthy(filename):
    # generate 24 months of transactions
    start_date = datetime(2024, 1, 1)
    end_date = datetime(2025, 12, 31)
    
    current_date = start_date
    current_balance = 250000.00  # strong starting balance
    transactions = []
    
    while current_date <= end_date:
        # super healthy business profile
        inflow_chance = 0.8 
        outflow_chance = 0.3
        
        # high revenue and controlled expenses
        inflow_amt = random.uniform(8000, 15000) 
        outflow_amt = random.uniform(4000, 9000)
        
        # inflow descriptions
        desc_in = random.choice([
            "Payment Gateway Settlement", 
            "E-commerce Sale Received", 
            "B2B Customer Deposit"
        ])
        
        # outflow descriptions
        desc_out = random.choice([
            "Wholesale Vendor Purchase", 
            "Logistics & Shipping Vendor", 
            "Staff Payroll", 
            "Office Rent", 
            "GST Tax Payment"
        ])

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
    
    # ensure correct column order
    df = df[['date', 'description', 'amount', 'balance']]
    df.to_csv(filename, index=False)
    
    # calculate summary stats
    total_in = df[df['description'].str.contains('Sale|Settlement|Deposit')]['amount'].sum()
    total_out = df[~df['description'].str.contains('Sale|Settlement|Deposit')]['amount'].sum()
    
    print(f" Generated {filename} with {len(df)} transactions.")
    print(f" Total Revenue: ₹{total_in:,.2f}")
    print(f" Total Expenses: ₹{total_out:,.2f}")
    print(f" Final Bank Balance: ₹{current_balance:,.2f}")

if __name__ == "__main__":
    generate_super_healthy("super_healthy_ecommerce.csv")