"""Slide 6 - How it works: the script runs again each time.

    streamlit run slides/slide06_reruns.py

Move the slider and watch the whole file execute again. The rerun counter at
the bottom is the only thing that survives it.
"""

import os
import pathlib

import pandas as pd
import streamlit as st

# ----------------------------------------------------------------- setup
# Not on the slide. Read customers.csv from the repo root, whichever
# directory this was launched from.
os.chdir(pathlib.Path(__file__).resolve().parent.parent)

# A label so the room knows which slide this belongs to.
st.caption("Slide 6 - How it works: the script runs again each time")


@st.cache_data
def load_data():
    return pd.read_csv("customers.csv")


# The slide's last line adds 1 to st.session_state.runs. Nothing creates it,
# so on the slide as printed that line raises AttributeError on first click.
if "runs" not in st.session_state:
    st.session_state.runs = 0

# ------------------------------------------------------- slide 6, verbatim
# reruns on every interaction
n = st.slider("Rows", 10, 100)
df = load_data()   # cached
st.dataframe(df.head(n))

if st.button("Recalculate"):
    st.session_state.runs += 1

# ----------------------------------------------------------------- after
st.caption(
    f"This script has run {st.session_state.get('script_runs', 0)} times. "
    f"The Recalculate button has been pressed {st.session_state.runs} times."
)
st.session_state.script_runs = st.session_state.get("script_runs", 0) + 1
