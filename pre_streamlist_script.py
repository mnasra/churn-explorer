import joblib
import pandas as pd

df = pd.read_csv("customers.csv")
bundle = joblib.load("model.pkl")
model, features = bundle["model"], bundle["features"]

region = "North"
subset = df[df["region"] == region]

print(f"Region: {region}")
print(f"Customers: {len(subset)}")
print(f"Churn rate: {subset['churned'].mean():.1%}")
print(subset.head(10))

case = pd.DataFrame(
    [{"tenure_months": 6, "monthly_spend": 85.0, "support_tickets": 3,
      "region": "North", "plan": "Basic"}]
)
print(f"Churn risk for that customer: {model.predict_proba(case[features])[0, 1]:.0%}")