import joblib

features = joblib.load("models/behaviour_feature_columns.pkl")
print(len(features))
print(features)