"""Slide 11 - Caching, so the app stays responsive.

    streamlit run slides/slide11_caching.py

Press Run again a few times with the toggle on, then turn it off and press it
a few more. Cached, only the first run pays for the work. Uncached, every
single one does, and the bar chart makes that hard to argue with.
"""

import os
import pathlib
import time

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st

# ----------------------------------------------------------------- setup
os.chdir(pathlib.Path(__file__).resolve().parent.parent)
DATA = "customers.csv"
PATH = "model.pkl"
BLUE, ORANGE = "#2a78d6", "#eb6834"
WINDOW = 14           # how many recent reruns the chart keeps on screen

st.caption("Slide 11 - Caching, so the app stays responsive")

# ------------------------------------------------------ slide 11, verbatim
@st.cache_data
def load_data(path):
    return pd.read_csv(path)


@st.cache_resource
def load_model():
    return joblib.load(PATH)


# ----------------------------------------------------------------- after
# Something slow enough to feel on a projector. This is real work, not a
# sleep: it really does build a 1.2 million row frame and group it.
def churn_summary(path):
    big = pd.concat([load_data(path)] * 300, ignore_index=True)
    big["band"] = pd.cut(big["tenure_months"], [0, 12, 24, 36, 48, 72])
    return (
        big.groupby(["region", "plan", "band"], observed=True)
        .agg(
            customers=("customer_id", "size"),
            churn_rate=("churned", "mean"),
            spend=("monthly_spend", "median"),
        )
        .reset_index()
    )


# The same function twice over. The only difference between them is the
# decorator, which is the whole point of the toggle below.
@st.cache_data
def summary_cached(path):
    return churn_summary(path)


def summary_uncached(path):
    return churn_summary(path)


left, right = st.columns([2, 1])
use_cache = left.toggle(
    "Wrap the summary in @st.cache_data", value=True,
    help="Off, the aggregation runs again on every single rerun.",
)
right.button("Run again", use_container_width=True)

# The first cached run after a clear is the cold one that fills the cache, so
# it is recorded separately and kept out of the comparison further down.
is_cold = use_cache and not st.session_state.get("primed", False)

started = time.perf_counter()
summary = summary_cached(DATA) if use_cache else summary_uncached(DATA)
elapsed_ms = (time.perf_counter() - started) * 1000

if use_cache:
    st.session_state.primed = True

# Every rerun appends to the log, so the contrast builds up on screen. The
# counter has to live in its own key: the log is trimmed to the last WINDOW
# entries, so counting its length would stop rising once it filled up.
if "history" not in st.session_state:
    st.session_state.history = []
st.session_state.run_no = st.session_state.get("run_no", 0) + 1
st.session_state.history.append(
    {
        "run": st.session_state.run_no,
        "ms": round(elapsed_ms, 1),
        "caching": "On" if use_cache else "Off",
        "cold": is_cold,
    }
)
st.session_state.history = st.session_state.history[-WINDOW:]

a, b, c = st.columns(3)
a.metric("This rerun", f"{elapsed_ms:,.0f} ms")
b.metric("Rows summarised", f"{len(load_data(DATA)) * 300:,}")
c.metric("Model AUC", f"{load_model()['auc']:.2f}",
         help="Loaded once by @st.cache_resource, however many times this reruns.")

history = pd.DataFrame(st.session_state.history)
history["label"] = [
    f"{ms:,.0f} cold" if cold else f"{ms:,.0f}"
    for ms, cold in zip(history["ms"], history["cold"])
]
fig = px.bar(
    history, x="run", y="ms", color="caching", template="plotly_white",
    color_discrete_map={"On": BLUE, "Off": ORANGE},
    category_orders={"caching": ["On", "Off"]},
    labels={"run": "Rerun", "ms": "Milliseconds", "caching": "Caching"},
    title="How long each rerun took",
    text="label",
)
fig.update_traces(textposition="outside", cliponaxis=False)
fig.update_layout(
    xaxis_dtick=1,
    legend=dict(orientation="h", y=1.02, yanchor="bottom", x=1, xanchor="right"),
)
st.plotly_chart(fig, use_container_width=True)

# Once both states have been tried, put the comparison into one sentence
# rather than making the room read it off the bars. The cold run is excluded
# by its own flag rather than by position, so it stays correct once the
# window has scrolled past it.
warm = history[(history["caching"] == "On") & (~history["cold"])]["ms"]
uncached = history[history["caching"] == "Off"]["ms"]
if len(warm) and len(uncached):
    st.success(
        f"Cached reruns take about {warm.median():,.0f} ms. Uncached, the same "
        f"work takes about {uncached.median():,.0f} ms - roughly "
        f"{uncached.median() / max(warm.median(), 0.1):,.0f} times longer, "
        "every time anyone touches a widget."
    )

st.caption(
    "Reading the CSV is cached either way by load_data above, so what the "
    "toggle controls is whether the aggregation on top of it reruns. Clearing "
    "the cache puts the next run back to cold."
)

if st.button("Clear caches"):
    st.cache_data.clear()
    st.cache_resource.clear()
    st.session_state.history = []
    st.session_state.run_no = 0
    st.session_state.primed = False
    st.rerun()

with st.expander("What the slow work actually produces"):
    st.dataframe(summary, use_container_width=True)
