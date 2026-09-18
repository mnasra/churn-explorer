
"""Customer churn explorer - the finished app for Demo 1 and Demo 2.

Run locally:   streamlit run app.py
Deploy:        push to GitHub, then connect the repo at share.streamlit.io
"""

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="Churn explorer", page_icon="📊", layout="wide")

DATA_PATH = "customers.csv"
MODEL_PATH = "model.pkl"


# Cached, so the file is read once rather than on every interaction.
@st.cache_data
def load_data(path):
    return pd.read_csv(path)


# cache_resource is for things you do not want copied: models, connections.
@st.cache_resource
def load_model(path):
    return joblib.load(path)


df = load_data(DATA_PATH)
bundle = load_model(MODEL_PATH)
model, features = bundle["model"], bundle["features"]

st.title("Customer churn explorer")
st.write(
    "Which customers are likely to leave, and why. Use the controls on the left "
    "to filter the base, or open the second tab to score a single customer."
)

# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("Filters")
    regions = st.multiselect("Region", sorted(df["region"].unique()),
                             default=sorted(df["region"].unique()))
    plans = st.multiselect("Plan", sorted(df["plan"].unique()),
                           default=sorted(df["plan"].unique()))
    max_tenure = st.slider("Maximum tenure (months)", 1, int(df["tenure_months"].max()),
                           int(df["tenure_months"].max()))

view = df[
    df["region"].isin(regions)
    & df["plan"].isin(plans)
    & (df["tenure_months"] <= max_tenure)
]

if view.empty:
    st.warning("Nothing matches those filters. Widen one of them on the left.")
    st.stop()

# ---------------------------------------------------------------- headline
c1, c2, c3 = st.columns(3)
c1.metric("Customers in view", f"{len(view):,}")
c2.metric("Churn rate", f"{view['churned'].mean():.1%}",
          delta=f"{view['churned'].mean() - df['churned'].mean():+.1%} vs all",
          delta_color="inverse")
c3.metric("Model AUC", f"{bundle['auc']:.2f}",
          help=f"Against a tenure-only baseline of {bundle['baseline']:.2f}")

overview, single, method = st.tabs(["Overview", "Score a customer", "Method"])

# ---------------------------------------------------------------- overview
with overview:
    left, right = st.columns(2)

    by_tenure = (
        view.assign(bucket=pd.cut(view["tenure_months"], bins=[0, 6, 12, 24, 48, 72]))
        .groupby("bucket", observed=True)["churned"].mean().reset_index()
    )
    by_tenure["bucket"] = by_tenure["bucket"].astype(str)
    fig1 = px.bar(by_tenure, x="bucket", y="churned",
                  labels={"bucket": "Tenure (months)", "churned": "Churn rate"},
                  title="Churn falls as customers stay longer")
    fig1.update_layout(yaxis_tickformat=".0%")
    left.plotly_chart(fig1, use_container_width=True)

    fig2 = px.scatter(view.sample(min(len(view), 800), random_state=1),
                      x="tenure_months", y="monthly_spend", color="churned",
                      labels={"tenure_months": "Tenure (months)",
                              "monthly_spend": "Monthly spend (£)",
                              "churned": "Churned"},
                      title="New customers on high spend are the risky corner")
    right.plotly_chart(fig2, use_container_width=True)

    with st.expander("See the underlying rows"):
        st.dataframe(view.head(200), use_container_width=True)

    st.download_button("Download this selection as CSV",
                       view.to_csv(index=False).encode(),
                       file_name="selection.csv", mime="text/csv")

# ---------------------------------------------------------------- scoring
with single:
    st.subheader("Score a single customer")
    a, b = st.columns(2)
    tenure = a.slider("Tenure (months)", 1, 72, 8)
    spend = a.slider("Monthly spend (£)", 8, 140, 85)
    tickets = b.slider("Support tickets", 0, 10, 3)
    region = b.selectbox("Region", sorted(df["region"].unique()))
    plan = b.selectbox("Plan", sorted(df["plan"].unique()))

    case = pd.DataFrame([{ "tenure_months": tenure, "monthly_spend": float(spend),
                           "support_tickets": tickets, "region": region, "plan": plan }])
    risk = model.predict_proba(case[features])[0, 1]

    st.metric("Churn risk", f"{risk:.0%}")
    st.progress(float(risk))
    if risk > 0.6:
        st.error("High risk. Worth a retention call.")
    elif risk > 0.35:
        st.warning("Moderate risk. Worth watching.")
    else:
        st.success("Low risk on these inputs.")
    st.caption(
        "A probability, not a verdict. The model has never seen this customer and "
        "knows nothing about them beyond the five fields above."
    )

# ---------------------------------------------------------------- method
with method:
    st.markdown(
        f"""
**Data.** {len(df):,} simulated customers, generated by `make_data.py`.
No real customer data is used anywhere in this demo.

**Model.** Gradient boosting on five features, one-hot encoding for region and
plan, trained in `train_model.py` and saved to `model.pkl`. The app only loads it.

**Performance.** AUC {bundle['auc']:.3f} on a 25% holdout, against a tenure-only
baseline of {bundle['baseline']:.3f}.

**Limitations.** The data is synthetic, so the relationships are cleaner than
anything real. There is no time-based validation, and no cost model behind the
risk thresholds above.
        """
    )