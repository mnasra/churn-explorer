# Slide companions

One runnable file per code slide, for showing the snippet working next to the
slide it comes from. Run any of them from the repo root:

    streamlit run slides/slide09_first_app.py

| Slide | File | Shows |
|---|---|---|
| 6 | `slide06_reruns.py` | The script runs top to bottom on every interaction; `st.session_state` is what survives it |
| 9 | `slide09_first_app.py` | A whole app in nine lines: title, data, widget, table, chart, with the selection feeding both |
| 10 | `slide10_widgets.py` | Each widget returns a value, shown live underneath |
| 11 | `slide11_caching.py` | A toggle that turns `@st.cache_data` on and off over the same real work, timing every rerun |
| 12 | `slide12_layout.py` | `st.columns`, `st.sidebar`, `st.tabs` |
| 13 | `slide13_charts.py` | A Plotly figure and a model probability side by side |

Every file is laid out the same way:

- a **setup** section with whatever the slide leaves undefined, clearly marked
  as not being on the slide
- the **slide's code, verbatim**, under its own banner
- sometimes an **after** section that makes the point visible

The slides are fragments, so most of them need a little scaffolding to run.
Two worth knowing about before anyone asks:

- **Slide 6** adds 1 to `st.session_state.runs` without ever creating it. As
  printed, that raises `AttributeError` the first time the button is pressed.
  The file initialises it in setup.
- **Slide 13** plots columns called `age` and `spend`. `customers.csv` has no
  age column, so the file renames tenure and spend to match. The axis label
  says `age` but the values are months of tenure.
