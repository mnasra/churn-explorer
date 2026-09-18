"""Slide 9 - A small app from start to finish.

    streamlit run slides/slide09_first_app.py

Nine lines: display, input, and a chart. The four labels on the slide -
Display, Input, Cache, Layout - all appear here except the cache, which is
slide 11.

Note that the selection feeds both the table and the chart. `view` is an
ordinary pandas DataFrame, and everything downstream reads from it.
"""

import os
import pathlib

import streamlit as st

# ----------------------------------------------------------------- setup
# Not on the slide. Read customers.csv from the repo root, whichever
# directory this was launched from. Streamlit is imported twice on purpose,
# so that the slide's own import line can stay where the slide puts it.
os.chdir(pathlib.Path(__file__).resolve().parent.parent)

# A label so the room knows which slide this belongs to.
st.caption("Slide 9 - A small app from start to finish")

# ------------------------------------------------------- slide 9, verbatim
import streamlit as st
import pandas as pd

st.title("Customer explorer")
df = pd.read_csv("customers.csv")
region = st.selectbox("Region", df.region.unique())
view = df[df.region == region]
st.dataframe(view)
st.bar_chart(view.groupby("plan").monthly_spend.sum())
