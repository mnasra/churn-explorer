
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

scores = model.predict_proba(X_test)[:, 1]

auc = roc_auc_score(y_test, scores)
baseline = roc_auc_score(y_test, X_test["tenure_months"].max() - X_test["tenure_months"])

print(f"Model AUC:    {auc:.3f}")
print(f"Tenure-only:  {baseline:.3f}   (the comparison baseline reported in the app)")

# The reference customer: the median of each numeric field and the mode of each
# categorical one. The app scores this row so that an individual score can be
# read against a benchmark rather than standing alone.
reference = {c: float(X_train[c].median()) for c in NUMERIC}
reference.update({c: X_train[c].mode()[0] for c in CATEGORICAL})

# The holdout goes into the bundle too. Choosing a threshold is a question
# about precision and recall, and those can only be read off data the model was
# not trained on. Scoring the training rows in the app would flatter it.
joblib.dump(
    {
        "model": model,
        "features": FEATURES,
        "auc": auc,
        "baseline": baseline,
        "reference": reference,
        "holdout_y": y_test.to_numpy(),
        "holdout_score": scores,
    },
    "model.pkl",
)
print(f"Wrote model.pkl ({len(y_test)} holdout rows saved for the threshold tab)")