import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, classification_report
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve
from sklearn.calibration import calibration_curve


df = pd.read_csv("behaviour_training_dataset.csv")

X = df.drop(columns=["default","company"])
y = df["default"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

model = RandomForestClassifier(
    n_estimators=250,
    max_depth=6,
    class_weight="balanced",
    random_state=42
)

model.fit(X_train, y_train)

pred = model.predict_proba(X_test)[:,1]

print("AUC:", roc_auc_score(y_test, pred))
print(classification_report(y_test, pred>0.5))

joblib.dump(model,"models/behaviour_risk_model.pkl")
joblib.dump(list(X.columns),"models/behaviour_feature_columns.pkl")

# create outputs folder if needed
os.makedirs("outputs", exist_ok=True)

print("\nGenerating evaluation graphs...")

# roc curve
fpr, tpr, _ = roc_curve(y_test, pred)

plt.figure()
plt.plot(fpr, tpr, label=f'AUC = {roc_auc_score(y_test, pred):.3f}')
plt.plot([0,1],[0,1],'--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.savefig("outputs/roc_curve.png")
plt.close()

# ks curve
data = pd.DataFrame({'y': y_test.values, 'p': pred}).sort_values('p')
data['cum_good'] = (1 - data['y']).cumsum() / (1 - data['y']).sum()
data['cum_bad'] = data['y'].cumsum() / data['y'].sum()

plt.figure()
plt.plot(data['cum_good'].values, label='Good (Non-Default)')
plt.plot(data['cum_bad'].values, label='Bad (Default)')
plt.title("KS Curve")
plt.legend()
plt.savefig("outputs/ks_curve.png")
plt.close()

# score distribution
import numpy as np
scores = np.clip(850 - (pred * 550), 300, 850)

plt.figure()
sns.histplot(scores[y_test==0], label='Good', kde=True, stat="density", color="blue")
sns.histplot(scores[y_test==1], label='Bad', kde=True, stat="density", color="red")
plt.legend()
plt.title("Score Distribution")
plt.savefig("outputs/score_distribution.png")
plt.close()

# calibration curve
prob_true, prob_pred = calibration_curve(y_test, pred, n_bins=10)

plt.figure()
plt.plot(prob_pred, prob_true, marker='o')
plt.plot([0,1],[0,1],'--')
plt.xlabel("Predicted PD")
plt.ylabel("Actual Default Rate")
plt.title("Calibration Curve")
plt.savefig("outputs/calibration_curve.png")
plt.close()

# feature importance
importance = pd.Series(model.feature_importances_, index=X.columns).sort_values()

plt.figure(figsize=(8,10))
importance.tail(15).plot(kind='barh')
plt.title("Top 15 Risk Drivers (Random Forest)")
plt.tight_layout()
plt.savefig("outputs/feature_importance.png")
plt.close()

print("Graphs successfully saved to the 'outputs/' folder.")