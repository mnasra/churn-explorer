"""Slide 13 - Charts and model output.

    streamlit run slides/slide13_charts.py

Pass a figure you built yourself to st.plotly_chart, and show the model's
confidence next to it rather than only the prediction.
"""

import os
import pathlib

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st

# ----------------------------------------------------------------- setup
os.chdir(pathlib.Path(__file__).resolve().parent.parent)

# The slide plots columns called "age" and "spend". customers.csv has no age
# column, so tenure and spend stand in under those names. Worth saying out
# loud if anyone reads the code closely.
df = pd.read_csv("customers.csv").rename(
    columns={"tenure_months": "age", "monthly_spend": "spend"}
).head(600)

# A label so the room knows which slide this belongs to.
st.caption("Slide 13 - Charts and model output")


def load_model():
    return joblib.load("model.pkl")["model"]


# One customer to score, in the column order the model was fitted on.
X = pd.DataFrame([{
    "tenure_months": 8,
    "monthly_spend": 85.0,
    "support_tickets": 3,
    "region": "East",
    "plan": "Basic",
}])

# ------------------------------------------------------ slide 13, verbatim
fig = px.scatter(df, x="age",
                 y="spend")
st.plotly_chart(fig)

model = load_model()
p = model.predict_proba(X)[0, 1]
st.metric("Churn risk", f"{p:.0%}")
