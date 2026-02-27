import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

def generate_super_healthy(filename):
    # Setup 24 months of dates
    start_date = datetime(2024, 1, 1)
    end_date = datetime(2025, 12, 31)
    
    current_date = start_date
    current_balance = 250000.00 # Starts with a very strong cash buffer
    transactions = []
    
    while current_date <= end_date:
        # SUPER HEALTHY PROFILE: Thriving E-Commerce / SaaS
        # 80% chance of making money every single day.
        # 30% chance of paying expenses.
        inflow_chance = 0.8 
        outflow_chance = 0.3
        
        # High daily revenue, strictly controlled expenses
        inflow_amt = random.uniform(8000, 15000) 
        outflow_amt = random.uniform(4000, 9000)
        
        # Keywords that match your Django views.py EXACTLY
        desc_in = random.choice([
            "Payment Gateway Settlement", 
            "E-commerce Sale Received", 
            "B2B Customer Deposit"
        ])
        
        desc_out = random.choice([
            "Wholesale Vendor Purchase", 
            "Logistics & Shipping Vendor", 
            "Staff Payroll", 
            "Office Rent", 
            "GST Tax Payment"
        ])

        # Generate Inflow (Cash In)
        if random.random() < inflow_chance:
            current_balance += inflow_amt
            transactions.append({
                "date": current_date.strftime("%Y-%m-%d"),
                "description": desc_in,
                "amount": round(inflow_amt, 2),
                "balance": round(current_balance, 2)
            })
            
        # Generate Outflow (Cash Out)
        if random.random() < outflow_chance:
            current_balance -= outflow_amt
            transactions.append({
                "date": current_date.strftime("%Y-%m-%d"),
                "description": desc_out,
                "amount": round(outflow_amt, 2),
                "balance": round(current_balance, 2)
            })
            
        current_date += timedelta(days=1)

    # Save to CSV
    df = pd.DataFrame(transactions)
    # Ensure columns match your Django expectation exactly
    df = df[['date', 'description', 'amount', 'balance']]
    df.to_csv(filename, index=False)
    
    # Calculate quick stats to prove how healthy it is
    total_in = df[df['description'].str.contains('Sale|Settlement|Deposit')]['amount'].sum()
    total_out = df[~df['description'].str.contains('Sale|Settlement|Deposit')]['amount'].sum()
    
    print(f"✅ Generated {filename} with {len(df)} transactions.")
    print(f"💰 Total Revenue: ₹{total_in:,.2f}")
    print(f"📉 Total Expenses: ₹{total_out:,.2f}")
    print(f"🏦 Final Bank Balance: ₹{current_balance:,.2f}")

if __name__ == "__main__":
    generate_super_healthy("super_healthy_ecommerce.csv")