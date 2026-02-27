import sys
import os
from pathlib import Path
import pandas as pd
import json
import io
from django.shortcuts import render, redirect
from django.contrib import messages
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.decorators import login_required
from .models import RawTransaction, CompanyCreditProfile
from .forms import FinancialUploadForm


# ==========================================
# AUTHENTICATION VIEWS
# ==========================================
def login_view(request):
    if request.user.is_authenticated:
        return redirect('upload_csv') # If already logged in, skip the login page

    if request.method == 'POST':
        u = request.POST.get('username')
        p = request.POST.get('password')
        user = authenticate(request, username=u, password=p)
        
        if user is not None:
            login(request, user)
            return redirect('upload_csv') # Success! Send them to the upload page
        else:
            messages.error(request, "Invalid credentials. Bank access denied.")
            
    return render(request, 'login.html')

def logout_view(request):
    logout(request)
    return redirect('login')
# ==========================================
# 1. THE UPLOAD VIEW (Now Supports Both CSV Formats)
# ==========================================
def upload_csv_view(request):
    if request.method == 'POST':
        form = FinancialUploadForm(request.POST, request.FILES)
        
        if form.is_valid():
            upload_instance = form.save()
            try:
                uploaded_file = request.FILES['file']
                file_name = uploaded_file.name.lower()
                uploaded_file.seek(0)
                
                if file_name.endswith('.csv'):
                    decoded_file = uploaded_file.read().decode('utf-8-sig')
                    df = pd.read_csv(io.StringIO(decoded_file), sep=',', engine='python')
                else:
                    df = pd.read_excel(uploaded_file)
                
                # Standardize all column names to lowercase for easy reading
                df.columns = df.columns.str.strip().str.lower()
                
                # Keywords to detect if a transaction is money coming IN
                credit_keywords = ["received", "deposit", "sale", "customer", "credit", "refund", "settlement"]
                
                transactions_to_create = []
                for _, row in df.iterrows():
                    # Handle the new 4-column demo_company format
                    if 'description' in df.columns:
                        desc = str(row['description'])
                        # Auto-detect if it's Cash In or Cash Out
                        txn_type_val = "CASH_IN" if any(k in desc.lower() for k in credit_keywords) else "CASH_OUT"
                        name_dest_val = desc # Save the raw description into the DB's nameDest field
                    
                    # Handle the old PaySim format just in case
                    else:
                        txn_type_val = row['type']
                        name_dest_val = row['namedest']

                    transactions_to_create.append(
                        RawTransaction(
                            nameOrig=upload_instance.business_name,
                            date=pd.to_datetime(row['date']),
                            txn_type=txn_type_val,
                            amount=row['amount'],
                            nameDest=name_dest_val,
                            balance=row['balance']
                        )
                    )
                
                RawTransaction.objects.bulk_create(transactions_to_create)
                return redirect('dashboard', business_name=upload_instance.business_name)
                
            except Exception as e:
                upload_instance.delete() 
                if 'df' in locals():
                    messages.error(request, f"Pandas crashed. Columns seen: {df.columns.tolist()} | Error: {str(e)}")
                else:
                    messages.error(request, f"File failed to open. Error: {str(e)}")
        else:
            print("FORM VALIDATION FAILED:", form.errors) # Prints to your terminal
            for field, errors in form.errors.items():
                for error in errors:
                    # Blasts the exact error to your UI as a red banner
                    messages.error(request, f"Upload Blocked - {field.title()}: {error}")
    else:
        form = FinancialUploadForm()
        
    return render(request, 'upload.html', {'form': form})


# ==========================================
# 2. THE DASHBOARD VIEW (AI & Smart Graphs)
# ==========================================
def dashboard_view(request, business_name):
    transactions = RawTransaction.objects.filter(nameOrig=business_name).order_by('date')
    
    if not transactions.exists():
        messages.error(request, "No transaction data found for this business.")
        return redirect('upload_csv')

    df = pd.DataFrame(list(transactions.values()))
    
    # =========================================================
    # --- AI PIPELINE INTEGRATION ---
    # =========================================================
    profile, created = CompanyCreditProfile.objects.get_or_create(
        business_name=business_name,
        defaults={'ml_features': {}} 
    )
    
    # FIX: Removed the "if created" check. Force the ML to run every time!
    try:
        WORKSPACE_ROOT = Path(__file__).resolve().parent.parent.parent.parent
        CREDIT_SCORING_DIR = WORKSPACE_ROOT / 'Credit-Scoring'
        sys.path.append(str(CREDIT_SCORING_DIR))
        
        from underwrite import underwrite_from_django
        
        ai_results = underwrite_from_django(df)
        
        profile.ml_features = ai_results["ml_features"]
        profile.credit_score = ai_results["Credit_Score"]
        profile.decision = ai_results["Decision"]
        profile.probability_of_default = ai_results["Probability_of_Default"]
        profile.risk_bucket = ai_results["Risk_Grade"]
        profile.top_risk_drivers = ai_results["Key_Risk_Drivers"]
        profile.save()
        
    except Exception as e:
        import traceback
        print(f"\n========== ML ERROR TRACE ==========\n{traceback.format_exc()}\n====================================\n")
        profile.credit_score = 0
        profile.decision = "PENDING"
        profile.top_risk_drivers = [f"Awaiting model sync. Error: {str(e)}"]
        profile.save()
            
    # =========================================================
    # --- REAL-WORLD GRAPHING LOGIC ---
    # =========================================================
    
    # 1. HIGH-LEVEL STATS (Strict Accounting)
    # Exclude "Opening Balance" so we don't artificially inflate revenue
    inflow_mask = (df['txn_type'] == 'CASH_IN') & (~df['nameDest'].str.contains('Opening Balance', case=False, na=False))
    outflow_mask = (df['txn_type'] == 'CASH_OUT')
    
    total_inflows_val = df[inflow_mask]['amount'].sum()
    total_outflows_val = df[outflow_mask]['amount'].sum()
    
    # --- CHART 1: Liquidity Trend (Running Balance) ---
    dates = df['date'].dt.strftime('%Y-%m-%d').tolist()
    balances = df['balance'].tolist()
    
    # --- CHART 2: Cash Velocity (24-Month Trend) ---
    # Group by Year-Month for long-term historical data
    df['chart_month'] = df['date'].dt.strftime('%Y-%m') 
    
    # Drop opening balance from the bar chart so it doesn't skew the visual scale
    flow_df = df[~df['nameDest'].str.contains('Opening Balance', case=False, na=False)]
    
    monthly_flow = flow_df.groupby(['chart_month', 'txn_type'])['amount'].sum().unstack(fill_value=0)
    
    if 'CASH_IN' not in monthly_flow.columns: monthly_flow['CASH_IN'] = 0
    if 'CASH_OUT' not in monthly_flow.columns: monthly_flow['CASH_OUT'] = 0
    
    velocity_labels = monthly_flow.index.tolist()
    inflows_list = monthly_flow['CASH_IN'].tolist()
    outflows_list = monthly_flow['CASH_OUT'].tolist()

    # --- CHART 3: STRICT Expense Categorization (Horizontal Bar Chart) ---
    def categorize_expense(desc):
        desc = str(desc).lower()
        if 'rent' in desc: return 'Rent'
        elif 'salary' in desc or 'staff' in desc: return 'Payroll'
        elif 'tax' in desc or 'gst' in desc: return 'Taxes'
        elif 'electricity' in desc or 'bill' in desc: return 'Utilities'
        elif 'vendor' in desc or 'purchase' in desc or 'wholesale' in desc: return 'Vendor Payments'
        else: return 'Other Expenses'

    # STRICT FIX: Only process OUTFLOWS.
    df_expenses = df[outflow_mask].copy()
    
    if not df_expenses.empty:
        df_expenses['category'] = df_expenses['nameDest'].apply(categorize_expense)
        # BUG FIX: Sum the actual money spent, don't just count the number of receipts!
        expense_sums = df_expenses.groupby('category')['amount'].sum().sort_values(ascending=False)
        txn_labels = expense_sums.index.tolist()
        txn_values = expense_sums.values.tolist()
    else:
        txn_labels = ["No Expenses"]
        txn_values = [0]
    # 3. Pack the Data for the Frontend
    context = {
        'business_name': business_name,
        
        # --- AI OUTPUTS ---
        'ml_score': profile.credit_score,
        'ml_decision': profile.decision,
        'pd_value': profile.probability_of_default,
        'decision_threshold': 0.35,
        'risk_bucket': profile.risk_bucket,
        'risk_drivers': profile.top_risk_drivers, 
        
        # High-level stats
        'avg_balance': f"₹ {df['balance'].mean():,.0f}",
        'total_inflow': f"₹ {total_inflows_val:,.0f}",
        'total_outflow': f"₹ {total_outflows_val:,.0f}",
        'txn_count': len(df),
        
        # JSON Data for Chart.js
        'dates_json': json.dumps(dates),
        'balances_json': json.dumps(balances),
        
        # Using velocity_labels mapped to the HTML's 'months_json' variable so you don't have to rewrite HTML
        'months_json': json.dumps(velocity_labels), 
        'inflows_json': json.dumps(inflows_list),
        'outflows_json': json.dumps(outflows_list),
        
        'txn_labels_json': json.dumps(txn_labels),
        'txn_values_json': json.dumps(txn_values),
    }
    
    return render(request, 'dashboard.html', context)
# ==========================================
# 3. MULTI-COMPANY PORTFOLIO VIEW
# ==========================================
def portfolio_view(request):
    # Fetch all AI profiles from the database, newest first
    profiles = CompanyCreditProfile.objects.all().order_by('-created_at')
    
    total_evaluations = profiles.count()
    approved_count = profiles.filter(decision='APPROVE').count()
    review_count = profiles.filter(decision='REVIEW').count()
    reject_count = profiles.filter(decision='REJECT').count()
    
    context = {
        'profiles': profiles,
        'total_evaluations': total_evaluations,
        'approved_count': approved_count,
        'review_count': review_count,
        'reject_count': reject_count,
    }
    
    return render(request, 'portfolio.html', context)
