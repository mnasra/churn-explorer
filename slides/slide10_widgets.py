"""Slide 10 - Adding inputs with widgets.

    streamlit run slides/slide10_widgets.py

Each widget returns a value. Assign it to a variable and use it like anything
else in Python. The panel underneath shows what each one is currently holding.
"""

import os
import pathlib

import pandas as pd
import streamlit as st

# ----------------------------------------------------------------- setup
# Not on the slide. The snippet uses a name `opts` that the slide never
# defines; these are the plan types in customers.csv.
os.chdir(pathlib.Path(__file__).resolve().parent.parent)
opts = sorted(pd.read_csv("customers.csv")["plan"].unique())

# A label so the room knows which slide this belongs to.
st.caption("Slide 10 - Adding inputs with widgets")

# ------------------------------------------------------ slide 10, verbatim
name = st.text_input("Name")
n = st.slider("Rows", 10, 500, 50)
cats = st.multiselect("Type", opts)
f = st.file_uploader("CSV")

if st.button("Run model"):
    st.success(f"Scored {n} rows")

# ----------------------------------------------------------------- after
st.divider()
st.write("What each widget is holding right now:")
st.json(
    {
        "name": name,
        "n": n,
        "cats": cats,
        "f": f.name if f else None,
    }
)
