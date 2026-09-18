"""Slide 12 - Laying out the page.

    streamlit run slides/slide12_layout.py

Three containers cover most cases: columns for things side by side, the
sidebar for controls, and tabs for detail kept one click away.
"""

import os
import pathlib

import pandas as pd
import streamlit as st

# ----------------------------------------------------------------- setup
# Not on the slide. Only needed to put something inside the tabs below.
os.chdir(pathlib.Path(__file__).resolve().parent.parent)
df = pd.read_csv("customers.csv")

# A label so the room knows which slide this belongs to.
st.caption("Slide 12 - Laying out the page")

# ------------------------------------------------------ slide 12, verbatim
c1, c2 = st.columns(2)
c1.metric("AUC", "0.87")
c2.metric("Rows", "12,480")

with st.sidebar:
    st.header("Controls")

a, b = st.tabs(["Model", "Data"])

with a:
    st.write("Whatever belongs on the model tab goes here.")

with b:
    st.dataframe(df.head(50), use_container_width=True)
