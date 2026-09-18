import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
N = 4000

regions = ["North", "South", "East", "West"]
plans = ["Basic", "Standard", "Premium"]

df = pd.DataFrame(
    {
        "customer_id": np.arange(1000, 1000 + N),
        "region": rng.choice(regions, N, p=[0.3, 0.3, 0.22, 0.18]),
        "plan": rng.choice(plans, N, p=[0.45, 0.35, 0.20]),
        "tenure_months": rng.integers(1, 72, N),
        "monthly_spend": np.round(rng.normal(48, 18, N).clip(8, 140), 2),
        "support_tickets": rng.poisson(1.1, N),
    }
)

# A churn signal the model can actually learn: short tenure, high spend,
# lots of support contact.
logit = (
    -4.2
    - 0.012 * df["tenure_months"]
    + 0.028 * df["monthly_spend"]
    + 0.50 * df["support_tickets"]
    + 0.0026 * df["monthly_spend"] * df["support_tickets"]
    + np.where(df["plan"] == "Basic", 1.3, np.where(df["plan"] == "Premium", -1.1, 0.0))
    + np.where(df["region"] == "West", 0.9, 0.0)
)
df["churned"] = (rng.random(N) < 1 / (1 + np.exp(-logit))).astype(int)

df.to_csv("customers.csv", index=False)
print(f"Wrote customers.csv: {len(df)} rows, churn rate {df.churned.mean():.1%}")