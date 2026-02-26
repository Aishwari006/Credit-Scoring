import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, classification_report
# from sklearn.preprocessing import KBinsDiscretizer
from sklearn.impute import SimpleImputer
# import category_encoders as ce

# ================================
# 1. LOAD DATA
# ================================

print("⏳ Loading MSME dataset...")
df = pd.read_excel(r"dataset\MSME Credit Data by 30S-CR.xlsx")

TARGET = "Label"

drop_cols = ["Enterprise_id", "Province", "Enterprise_type", "Sector"]
X = df.drop(columns=drop_cols + [TARGET])
X = X.select_dtypes(include=[np.number])
X = X.sort_index(axis=1)
y = df[TARGET]


# ================================
# 2. TRAIN TEST SPLIT
# ================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

X_train = X_train.reset_index(drop=True)
X_test = X_test.reset_index(drop=True)
y_train = y_train.reset_index(drop=True)
y_test = y_test.reset_index(drop=True)

# ================================
# 3. IMPUTATION (NO LEAKAGE)
# ================================

# ================================
# 3. IMPUTATION (NO LEAKAGE)
# ================================
imputer = SimpleImputer(strategy="median")

X_train = pd.DataFrame(imputer.fit_transform(X_train), columns=X_train.columns)
X_test = pd.DataFrame(imputer.transform(X_test), columns=X_train.columns)

# protect against infinite values from ratios
X_train = X_train.replace([np.inf, -np.inf], np.nan)
X_test = X_test.replace([np.inf, -np.inf], np.nan)

X_train = X_train.fillna(X_train.median())
X_test = X_test.fillna(X_train.median())

# ================================
# 4. SCALING (robust for outliers)
# ================================
from sklearn.preprocessing import RobustScaler

scaler = RobustScaler()

X_train_scaled = pd.DataFrame(
    scaler.fit_transform(X_train),
    columns=X_train.columns
)

X_test_scaled = pd.DataFrame(
    scaler.transform(X_test),
    columns=X_train.columns
)

# ================================
# 5. UNIVARIATE AUC FEATURE RANKING (Top-N Selection)
# ================================
from sklearn.metrics import roc_auc_score

auc_scores = []

for col in X_train_scaled.columns:
    try:
        auc = roc_auc_score(y_train, X_train_scaled[col])
        auc_scores.append((col, max(auc, 1-auc)))
    except:
        pass

auc_df = pd.DataFrame(auc_scores, columns=['feature','auc'])
auc_df = auc_df.sort_values('auc', ascending=False)

selected = auc_df.head(25)['feature'].tolist()

# enforce fixed column order
model_input_columns = selected

X_train_scaled = X_train_scaled[model_input_columns]
X_test_scaled = X_test_scaled[model_input_columns]

print("Features after AUC ranking:", len(model_input_columns))
print(auc_df.head(10))

# save schema ONCE (correct location)
joblib.dump(model_input_columns, "05_model_input_columns.pkl")

# save fallback values for production scoring
feature_defaults = X_train_scaled.median().to_dict()
joblib.dump(feature_defaults, "08_feature_defaults.pkl")



# ================================
# 6. LOGISTIC SCORECARD MODEL
# ================================
base_model = LogisticRegression(
    penalty='elasticnet',
    solver='saga',
    l1_ratio=0.6,
    C=0.5,
    class_weight='balanced',
    max_iter=5000
)

# CV stability check
from sklearn.model_selection import StratifiedKFold, cross_val_score
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

cv_auc = cross_val_score(
    base_model,
    X_train_scaled,
    y_train,
    scoring='roc_auc',
    cv=cv
)

print("\n📊 MODEL STABILITY (5-fold CV on selected features)")
print("Stability AUC Mean:", round(cv_auc.mean(),3))
print("Stability AUC Std:", round(cv_auc.std(),3))

# fit interpretable model
base_model.fit(X_train_scaled, y_train)

# calibrated deployment model
from sklearn.calibration import CalibratedClassifierCV
deployed_model = CalibratedClassifierCV(
    LogisticRegression(
        penalty='elasticnet',
        solver='saga',
        l1_ratio=0.6,
        C=0.5,
        class_weight='balanced',
        max_iter=5000
    ),
    method='isotonic',
    cv=5
)

deployed_model.fit(X_train_scaled, y_train)



def ks_stat(y_true, y_prob):
    data = pd.DataFrame({'y':y_true.values,'p':y_prob})
    data = data.sort_values('p')
    data['cum_good'] = (1-data['y']).cumsum()/(1-data['y']).sum()
    data['cum_bad'] = data['y'].cumsum()/data['y'].sum()
    return np.max(np.abs(data['cum_bad']-data['cum_good']))

# ================================
# Learn decision threshold on TRAIN (prevents test leakage)
# ================================
train_proba = deployed_model.predict_proba(X_train_scaled)[:,1]

thresholds = np.linspace(0.01,0.5,200)
ks_values = []

for t in thresholds:
    mask = train_proba >= t
    if mask.sum() == 0 or mask.sum() == len(train_proba):
        ks_values.append(0)
    else:
        ks_values.append(abs(y_train[mask].mean() - y_train[~mask].mean()))

best_threshold = thresholds[np.argmax(ks_values)]
print("Chosen Threshold (from training):", round(best_threshold,3))

# ================================
# 7. MODEL VALIDATION
# ================================
proba = deployed_model.predict_proba(X_test_scaled)[:,1]
# ---------------------------------
# Optimal threshold using KS
# ---------------------------------
pred = (proba >= best_threshold).astype(int)


print("\n📊 MODEL PERFORMANCE")
print("AUROC:", round(roc_auc_score(y_test, proba),3))
from sklearn.metrics import brier_score_loss
print("Brier Score:", round(brier_score_loss(y_test, proba),4))
print("KS:", round(ks_stat(y_test, proba),3))
print(classification_report(y_test, pred))
top_decile = np.percentile(proba,90)
capture = y_test[proba>=top_decile].mean()
print("Top 10% risk default rate:", round(capture,3))

# ================================
# 8. CREDIT SCORE SCALING (300-900)
# ================================
PDO = 50
BASE_SCORE = 600
BASE_ODDS = 2

factor = PDO / np.log(2)
offset = BASE_SCORE - factor*np.log(BASE_ODDS)

scorecard_scaling = {
    "PDO": PDO,
    "BASE_SCORE": BASE_SCORE,
    "BASE_ODDS": BASE_ODDS,
    "factor": factor,
    "offset": offset
}

def prob_to_score(p):
    p = np.clip(p, 1e-6, 1-1e-6)
    odds = p/(1-p)
    return offset - factor*np.log(odds)

scores = np.clip(prob_to_score(proba),300,900)
print("\nExample Scores:", scores[:10])

print("Average score:", round(scores.mean(),1))

# ================================
# 9. EXPLAINABILITY (REASON CODES)
# ================================
coefficients = pd.Series(base_model.coef_[0], index=X_train_scaled.columns)

feature_map = {
    # Distress & Indiscipline
    "Overdue_tax_num_1year": "Recent severe cash flow shortage (Overdue Taxes)",
    "Executee_num_1year": "Recent legal enforcement actions against the business",
    "Legal_proceedings_num_1year": "Active legal litigation indicating financial distress",
    "Administrative_penalty_num_1year": "Recent regulatory penalties indicating poor operational discipline",
    
    # Scale & Liquidity
    "Registered_capital (Ten thousand Yuan)": "Insufficient capital buffer to absorb financial shocks",
    "Paid_in_capital (Ten thousand Yuan)": "Low paid-in liquidity reserves",
    "Establishment_Duration (Days)": "Insufficient operational history (Thin-file risk)",
    
    # Payroll / Stability Proxies
    "t-1 Basic medical insurance for employees": "Drop in payroll consistency (Instability in working capital)",
    "t-1 Basic old-age insurance for urban employees": "Inconsistent employee benefit payments",
    
    # Positive signals (if their lack causes a rejection)
    "Ratepaying_Credit_Grade_A_num": "Lack of verified high-tier tax compliance history"
}

def explain_applicant(row_scaled, coeffs, top_n=3):
    contrib = row_scaled * coeffs
    top = contrib.sort_values(ascending=False).head(top_n)
    reasons=[]
    for f,v in top.items():
        if v>0:
            reasons.append(feature_map.get(f,f))
    return reasons

print("\n🧠 Example Explanation:")
print(explain_applicant(X_test_scaled.iloc[0], coefficients))

# ================================
# 10. SAVE FULL PIPELINE
# ================================
joblib.dump(imputer,"01_imputer.pkl")
joblib.dump(scaler,"02_scaler.pkl")
joblib.dump(deployed_model,"03_scorecard_model.pkl")
joblib.dump(scorecard_scaling, "07_scorecard_scaling.pkl")
joblib.dump(best_threshold, "06_decision_threshold.pkl")

print("\n✅ Training Complete — Scorecard Ready for Deployment")

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from sklearn.calibration import calibration_curve
import seaborn as sns

# =========================
# ROC CURVE
# =========================
fpr, tpr, _ = roc_curve(y_test, proba)

plt.figure()
plt.plot(fpr, tpr, label=f'AUC = {roc_auc_score(y_test, proba):.3f}')
plt.plot([0,1],[0,1],'--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.savefig("roc_curve.png")
plt.close()

# =========================
# KS CURVE
# =========================
data = pd.DataFrame({'y':y_test.values,'p':proba}).sort_values('p')
data['cum_good'] = (1-data['y']).cumsum()/(1-data['y']).sum()
data['cum_bad'] = data['y'].cumsum()/data['y'].sum()

plt.figure()
plt.plot(data['cum_good'], label='Good')
plt.plot(data['cum_bad'], label='Bad')
plt.title("KS Curve")
plt.legend()
plt.savefig("ks_curve.png")
plt.close()

# =========================
# SCORE DISTRIBUTION
# =========================
scores = np.clip(prob_to_score(proba),300,900)

plt.figure()
sns.histplot(scores[y_test==0], label='Good', kde=True, stat="density")
sns.histplot(scores[y_test==1], label='Bad', kde=True, color='red', stat="density")
plt.legend()
plt.title("Score Distribution")
plt.savefig("score_distribution.png")
plt.close()

# =========================
# CALIBRATION CURVE (PD reliability)
# =========================
prob_true, prob_pred = calibration_curve(y_test, proba, n_bins=10)

plt.figure()
plt.plot(prob_pred, prob_true, marker='o')
plt.plot([0,1],[0,1],'--')
plt.xlabel("Predicted PD")
plt.ylabel("Actual Default Rate")
plt.title("Calibration Curve")
plt.savefig("calibration_curve.png")
plt.close()

# =========================
# FEATURE IMPORTANCE (Explainability)
# =========================
importance = pd.Series(base_model.coef_[0], index=X_train_scaled.columns).sort_values()

plt.figure(figsize=(8,10))
importance.tail(15).plot(kind='barh')
plt.title("Top Risk Drivers")
plt.savefig("feature_importance.png")
plt.close()

print("\n📈 Graphs saved:")
print("roc_curve.png")
print("ks_curve.png")
print("score_distribution.png")
print("calibration_curve.png")
print("feature_importance.png")