
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

NUMERIC = ["tenure_months", "monthly_spend", "support_tickets"]
CATEGORICAL = ["region", "plan"]
FEATURES = NUMERIC + CATEGORICAL

df = pd.read_csv("customers.csv")
X_train, X_test, y_train, y_test = train_test_split(
    df[FEATURES], df["churned"], test_size=0.25, random_state=42, stratify=df["churned"]
)

model = Pipeline(
    [
        (
            "prep",
            ColumnTransformer(
                [("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL)],
                remainder="passthrough",
            ),
        ),
        ("clf", GradientBoostingClassifier(random_state=42)),
    ]
)

model.fit(X_train, y_train)

auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
baseline = roc_auc_score(y_test, X_test["tenure_months"].max() - X_test["tenure_months"])

print(f"Model AUC:    {auc:.3f}")
print(f"Tenure-only:  {baseline:.3f}   (the baseline worth quoting on a slide)")

joblib.dump({"model": model, "features": FEATURES, "auc": auc, "baseline": baseline}, "model.pkl")
print("Wrote model.pkl")