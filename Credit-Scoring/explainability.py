import requests

def generate_explanation(features_row):
    # 1. DETERMINISTIC RULE ENGINE (Ground Truth)
    # These become your 1-line pointwise explanations
    base_reasons = []

    if features_row.get("balance_cv", 0) > 1.0:
        base_reasons.append("Highly volatile cash balance")
    if features_row.get("debit_credit_ratio", 0) > 0.9:
        base_reasons.append("Dangerously high expense-to-income ratio")
    if features_row.get("fixed_obligation_ratio", 0) > 0.4:
        base_reasons.append("Heavy burden of recurring financial commitments")
    if features_row.get("sudden_drop_events", 0) > 0:
        base_reasons.append("History of severe balance drop events")
    if features_row.get("months_without_credit", 0) > 0:
        base_reasons.append("Periods with absolutely zero incoming revenue")
    if features_row.get("top_counterparty_share", 0) > 0.6:
        base_reasons.append("Dangerous reliance on a single primary customer")
    if features_row.get("negative_balance_days", 0) > 0:
        base_reasons.append("Account repeatedly enters overdraft/negative balance")

    if not base_reasons:
        base_reasons.append("Stable financials with no severe distress markers")

    # 2. CONSTRUCT THE AGENTIC PROMPT
    prompt = f"""
    You are a strict, senior credit risk underwriter for a fintech company. 
    Analyze this MSME business based on the following algorithmic triggers and metrics.
    
    Triggered Risk Factors: {', '.join(base_reasons)}
    
    Key Metrics:
    - Average Daily Balance: {features_row.get('avg_balance', 0):.2f}
    - Expense-to-Income Ratio: {features_row.get('debit_credit_ratio', 0):.2f}
    - Low Balance Days: {features_row.get('low_balance_days', 0)}
    - Total Sudden Drop Events: {features_row.get('sudden_drop_events', 0)}
    
    Write a cohesive, professional 3-sentence risk summary explaining WHY this company is either a safe bet or a dangerous risk. 
    Do not use bullet points. Do not introduce yourself. Just provide the analytical paragraph.
    """

    # 3. CALL LOCAL OLLAMA API
    try:
        response = requests.post('http://localhost:11434/api/generate', json={
            "model": "qwen3:8b",
            "prompt": prompt,
            "stream": False
        }, timeout=60) 

        if response.status_code == 200:
            llm_text = response.json()['response'].strip()
            
            # ==========================================
            # THE FIX: COMBINE POINTS AND PARAGRAPH
            # ==========================================
            # We add a bold prefix so the paragraph stands out visually from the points
            detailed_paragraph = f"AI Deep Dive: {llm_text}"
            
            # Return the bullet points first, then the detailed paragraph at the bottom
            return base_reasons + [detailed_paragraph]
            
        else:
            print("Ollama returned an error, falling back to static rules.")
            return base_reasons
            
    except requests.exceptions.RequestException as e:
        # HACKATHON SAFETY NET
        print(f"Ollama connection failed: {e}. Falling back to static list.")
        return base_reasons