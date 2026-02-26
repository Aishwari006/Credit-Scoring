import joblib

features = joblib.load("08_feature_defaults.pkl")
print(len(features))
print(features)