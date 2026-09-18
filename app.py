"""Customer churn explorer - the finished app for Demo 1 and Demo 2.

Run locally:   streamlit run app.py
Deploy:        push to GitHub, then connect the repo at share.streamlit.io
"""

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="Churn explorer", page_icon="📊", layout="wide")

DATA_PATH = "customers.csv"
MODEL_PATH = "model.pkl"

# One place for the colours, so every chart on the page agrees with the others.
BLUE, ORANGE = "#2a78d6", "#eb6834"
RAISES, LOWERS = "#e34948", "#2a78d6"
TEMPLATE = "plotly_white"

# Bands for the overview charts, and the smallest group worth quoting a rate for.
SPEND_BINS = [0, 30, 45, 60, 75, 140]
SPEND_LABELS = ["under 30", "30-45", "45-60", "60-75", "over 75"]
TENURE_BINS = [0, 12, 24, 36, 48, 72]
TENURE_LABELS = ["under 12", "12-24", "24-36", "36-48", "over 48"]
MIN_CELL = 30

LABELS = {
    "tenure_months": "Tenure (months)",
    "monthly_spend": "Monthly spend (£)",
    "support_tickets": "Support tickets",
    "region": "Region",
    "plan": "Plan",
}


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

# If someone clones the repo and runs the app against an older model file, say
# so plainly rather than failing with a KeyError halfway down the page.
if not {"holdout_y", "reference"} <= bundle.keys():
    st.error(
        "`model.pkl` was written by an earlier version of `train_model.py` and does "
        "not contain holdout scores. Re-run `python train_model.py`, then reload."
    )
    st.stop()

model = bundle["model"]
FEATURES = bundle["features"]
reference = bundle["reference"]
holdout_y = bundle["holdout_y"]
holdout_score = bundle["holdout_score"]

st.title("Customer churn explorer")
st.write(
    "Churn risk across the customer base, the factors behind an individual score, "
    "and the trade-offs involved in choosing a contact threshold. The threshold "
    "control in the sidebar applies to every tab."
)


# ------------------------------------------------------------------ helpers
def score_one(values):
    """Churn risk for a single customer, as a probability."""
    return float(model.predict_proba(pd.DataFrame([values])[FEATURES])[0, 1])


def what_moves_this(case):
    """How much of this customer's risk comes from each field.

    Swap one field at a time for the reference customer's value and score it
    again. The gap that opens up is what that one field is worth, holding the
    rest of the customer fixed.
    """
    actual = score_one(case)
    rows = []
    for field in FEATURES:
        rows.append(
            {
                "field": LABELS[field],
                "effect": actual - score_one({**case, field: reference[field]}),
                "theirs": case[field],
                "reference": reference[field],
            }
        )
    return pd.DataFrame(rows).sort_values("effect")


@st.cache_data
def threshold_table(y, score, call_cost, save_value, save_rate):
    """Precision, recall and net value at every threshold, on the holdout.

    Cached because it is the same arithmetic on every rerun, and only the three
    cost assumptions ever change it.
    """
    per_1000 = 1000 / len(y)
    rows = []
    for t in np.round(np.arange(0.05, 0.96, 0.01), 2):
        flagged = score >= t
        n = int(flagged.sum())
        caught = int((flagged & (y == 1)).sum())
        rows.append(
            {
                "threshold": t,
                "flagged": n,
                "precision": caught / n if n else np.nan,
                "recall": caught / y.sum(),
                "net_value": (caught * save_rate * save_value - n * call_cost) * per_1000,
            }
        )
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.header("Filters")
    regions = st.multiselect(
        "Region", sorted(df["region"].unique()), default=sorted(df["region"].unique())
    )
    plans = st.multiselect(
        "Plan", sorted(df["plan"].unique()), default=sorted(df["plan"].unique())
    )
    max_tenure = st.slider(
        "Maximum tenure (months)", 1, int(df["tenure_months"].max()),
        int(df["tenure_months"].max()),
    )

    st.divider()
    st.header("Decision threshold")
    # Sliders in whole percent rather than 0-1: the label reads "50%" instead of
    # "0.5", and there is no float formatting to get wrong.
    threshold = st.slider(
        "Flag customers above", 5, 95, 50, 1, format="%d%%",
        help="Risk score above which a customer is added to the retention list.",
    ) / 100

    with st.expander("Cost assumptions"):
        call_cost = st.number_input("Cost per retention contact (£)", 1, 200, 25)
        save_value = st.number_input("Value of a retained customer (£)", 50, 5000, 400, step=50)
        save_rate = st.slider("Contact success rate", 5, 80, 30, 5, format="%d%%") / 100

view = df[
    df["region"].isin(regions)
    & df["plan"].isin(plans)
    & (df["tenure_months"] <= max_tenure)
]

if view.empty:
    st.warning(
        "No customers match the current filters. At least one selection is "
        "required in each sidebar filter."
    )
    st.stop()

curve = threshold_table(holdout_y, holdout_score, call_cost, save_value, save_rate)
here = curve.iloc[(curve["threshold"] - threshold).abs().idxmin()]
best = curve.loc[curve["net_value"].idxmax()]

# ----------------------------------------------------------------- headline
c1, c2, c3 = st.columns(3)
c1.metric("Customers in view", f"{len(view):,}")
gap = view["churned"].mean() - df["churned"].mean()
c2.metric(
    "Churn rate", f"{view['churned'].mean():.1%}",
    # No arrow when nothing is filtered out, rather than a red "+0.0%".
    delta=f"{gap:+.1%} vs all customers" if len(view) < len(df) else None,
    delta_color="inverse",
)
c3.metric(
    "Model AUC", f"{bundle['auc']:.2f}",
    delta=f"{bundle['auc'] - bundle['baseline']:+.2f} vs. tenure-only baseline",
    help=f"A tenure-only ranking scores {bundle['baseline']:.2f} on the same holdout.",
)

overview, single, decision, method = st.tabs(
    ["Overview", "Score a customer", "Decision threshold", "Method"]
)

# ----------------------------------------------------------------- overview
with overview:
    st.write(
        "Monthly spend separates this base more sharply than tenure does, which "
        "accounts for the model's advantage over a tenure-only ranking."
    )
    left, right = st.columns(2)

    # Churn by spend band, next to churn by tenure band, on the same axis so the
    # two are directly comparable. Spend is the one that moves.
    banded = view.assign(
        spend_band=pd.cut(view["monthly_spend"], SPEND_BINS, labels=SPEND_LABELS),
        tenure_band=pd.cut(view["tenure_months"], TENURE_BINS, labels=TENURE_LABELS),
    )
    bars = pd.concat(
        [
            banded.groupby("spend_band", observed=True)["churned"].agg(["mean", "size"])
            .rename_axis("band").assign(driver="Monthly spend (£)"),
            banded.groupby("tenure_band", observed=True)["churned"].agg(["mean", "size"])
            .rename_axis("band").assign(driver="Tenure (months)"),
        ]
    ).reset_index()
    bars = bars[bars["size"] >= MIN_CELL]

    # Where the two meet. Cells with too few customers behind them are left
    # blank rather than shown as a rate nobody should act on.
    grid = banded.pivot_table(
        index="spend_band", columns="tenure_band", values="churned",
        aggfunc="mean", observed=False,
    )
    counts = banded.pivot_table(
        index="spend_band", columns="tenure_band", values="churned",
        aggfunc="size", observed=False,
    ).reindex_like(grid).fillna(0)
    grid = grid.where(counts >= MIN_CELL)

    # Under a narrow enough filter every band falls below the minimum. Report
    # that, rather than drawing empty axes or raising on an empty max().
    if bars.empty:
        left.info(
            f"No band retains at least {MIN_CELL} customers under the current "
            "filters. Widen the sidebar filters to display this chart."
        )
    else:
        fig1 = px.bar(
            bars, x="band", y="mean", facet_col="driver", template=TEMPLATE,
            labels={"band": "", "mean": "Churn rate"},
            title="Churn rate by spend, and by tenure",
            text=bars["mean"].map("{:.0%}".format),
            custom_data=["size"],
        )
        fig1.update_traces(
            marker_color=BLUE, textposition="outside", cliponaxis=False,
            hovertemplate="%{x}<br>Churn rate %{y:.1%}"
                          "<br>%{customdata[0]:,} customers<extra></extra>",
        )
        fig1.update_xaxes(matches=None, showticklabels=True)
        fig1.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
        fig1.update_layout(
            yaxis_tickformat=".0%", yaxis_range=[0, bars["mean"].max() * 1.25],
            showlegend=False, margin=dict(t=70),
        )
        left.plotly_chart(fig1, use_container_width=True)

    if grid.isna().all().all():
        right.info(
            "Every cell of the spend-by-tenure grid falls below the minimum base "
            "under the current filters."
        )
    else:
        fig2 = px.imshow(
            grid, template=TEMPLATE, aspect="auto", origin="lower",
            color_continuous_scale=[[0, "#cde2fb"], [0.5, "#2a78d6"], [1, "#0d366b"]],
            labels={"x": "Tenure (months)", "y": "Monthly spend (£)",
                    "color": "Churn rate"},
            title="Churn rate by spend and tenure",
            text_auto=".0%",
        )
        fig2.update_traces(
            xgap=2, ygap=2, customdata=counts.where(counts >= MIN_CELL).to_numpy(),
            hovertemplate="£%{y} a month, %{x} months<br>Churn rate %{z:.1%}"
                          "<br>%{customdata:,.0f} customers<extra></extra>",
        )
        fig2.update_layout(
            coloraxis_colorbar=dict(tickformat=".0%", title=""), margin=dict(t=70)
        )
        right.plotly_chart(fig2, use_container_width=True)
        right.caption(
            f"Cells below {MIN_CELL} customers are left blank. The top spend band "
            "remains thin at roughly 50 customers per cell, and is better read as a "
            "single band than as five separate rates. Hover for the base behind "
            "any cell."
        )

    with st.expander("Underlying rows"):
        st.dataframe(view.head(200), use_container_width=True)

    st.download_button(
        "Download this selection as CSV",
        view.to_csv(index=False).encode(),
        file_name="selection.csv", mime="text/csv",
    )

# ------------------------------------------------------------------ scoring
with single:
    st.subheader("Score a single customer")
    a, b = st.columns(2)
    tenure = a.slider("Tenure (months)", 1, 72, 8)
    spend = a.slider("Monthly spend (£)", 8, 140, 85)
    tickets = b.slider("Support tickets", 0, 10, 3)
    region = b.selectbox("Region", sorted(df["region"].unique()))
    plan = b.selectbox("Plan", sorted(df["plan"].unique()))

    case = {
        "tenure_months": tenure, "monthly_spend": float(spend),
        "support_tickets": tickets, "region": region, "plan": plan,
    }
    risk = score_one(case)
    reference_risk = score_one(reference)

    result, chart = st.columns([1, 2])
    with result:
        st.metric(
            "Churn risk", f"{risk:.0%}",
            delta=f"{risk - reference_risk:+.0%} vs. the reference customer",
            delta_color="inverse",
        )
        st.progress(float(risk))
        if risk >= threshold:
            st.error(f"Above the {threshold:.0%} threshold. Flagged for contact.")
        else:
            st.success(f"Below the {threshold:.0%} threshold. Not flagged.")
        st.caption(
            f"Reference customer: {reference['tenure_months']:.0f} months tenure, "
            f"£{reference['monthly_spend']:.0f} per month, "
            f"{reference['support_tickets']:.0f} support ticket"
            f"{'' if reference['support_tickets'] == 1 else 's'}. "
            f"Scores {reference_risk:.0%}."
        )

    with chart:
        contrib = what_moves_this(case)
        contrib["direction"] = np.where(contrib["effect"] > 0, "Raises risk", "Lowers risk")
        fig3 = px.bar(
            contrib, x="effect", y="field", orientation="h", template=TEMPLATE,
            color="direction",
            color_discrete_map={"Raises risk": RAISES, "Lowers risk": LOWERS},
            labels={"effect": "Change in risk", "field": "", "direction": ""},
            title="Contribution of each field to this score",
            # "0%" rather than "-0%" where a field matches the reference customer.
            text=contrib["effect"].map(lambda v: "0%" if abs(v) < 0.005 else f"{v:+.0%}"),
            custom_data=["theirs", "reference"],
        )
        fig3.update_traces(
            textposition="outside", cliponaxis=False,
            hovertemplate="%{y}<br>This customer: %{customdata[0]}"
                          "<br>Reference: %{customdata[1]}"
                          "<br>Effect %{x:+.1%}<extra></extra>",
        )
        fig3.update_layout(
            xaxis_tickformat=".0%", height=330, bargap=0.45,
            margin=dict(l=0, r=0, t=60, b=0),
            legend=dict(orientation="h", y=1.02, yanchor="bottom", x=1, xanchor="right"),
        )
        st.plotly_chart(fig3, use_container_width=True)
        st.caption(
            "Each bar shows the change in risk when that field alone is set to the "
            "reference customer's value. The bars do not sum to the total score, "
            "because the model allows the fields to interact."
        )

    st.caption(
        "The output is a probability, not a decision. The model has not seen this "
        "customer and uses only the five fields above."
    )

# ---------------------------------------------------------------- threshold
with decision:
    st.subheader("Setting the decision threshold")
    st.write(
        "Converting scores into a contact list requires a cut-off. That choice is "
        "commercial rather than statistical: a high threshold misses churners, and a "
        "low one spends the recovered value contacting customers who would have "
        "stayed. All figures below are measured on the 25% holdout."
    )

    random_precision = float(holdout_y.mean())
    d1, d2, d3, d4 = st.columns(4)
    d1.metric("Customers flagged", f"{here['flagged'] / len(holdout_y):.0%}")
    d2.metric(
        "Precision", f"{here['precision']:.0%}",
        delta=f"{here['precision'] / random_precision:.1f}x random selection",
        help=f"Random selection would be correct {random_precision:.0%} of the time.",
    )
    d3.metric("Churners identified", f"{here['recall']:.0%}")
    behind = here["net_value"] - best["net_value"]
    d4.metric(
        "Net value", f"£{here['net_value']:,.0f}",
        # Built sign-first, so a shortfall shows as a red fall rather than a
        # green rise: Streamlit reads the leading character, not the currency.
        delta=("at the optimum" if behind == 0
               else f"{'-' if behind < 0 else '+'}£{abs(behind):,.0f} vs. the optimum"),
        help="Per 1,000 customers, under the cost assumptions in the sidebar.",
    )

    left, right = st.columns(2)

    tidy = curve.melt(
        id_vars="threshold", value_vars=["precision", "recall"],
        var_name="measure", value_name="rate",
    ).replace({"precision": "Precision", "recall": "Churners identified"})
    fig4 = px.line(
        tidy, x="threshold", y="rate", color="measure", template=TEMPLATE,
        color_discrete_map={"Precision": BLUE, "Churners identified": ORANGE},
        labels={"threshold": "Threshold", "rate": "Rate", "measure": ""},
        title="Precision and recall by threshold",
    )
    fig4.update_traces(line_width=2)
    fig4.add_vline(x=threshold, line_dash="dot", line_color="#52514e",
                   annotation_text="current", annotation_position="top left")
    fig4.update_layout(
        xaxis_tickformat=".0%", yaxis_tickformat=".0%", yaxis_title="Rate",
        legend=dict(orientation="h", y=1.02, yanchor="bottom", x=1, xanchor="right"),
    )
    left.plotly_chart(fig4, use_container_width=True)

    fig5 = px.line(
        curve, x="threshold", y="net_value", template=TEMPLATE,
        labels={"threshold": "Threshold", "net_value": "Net value per 1,000 customers (£)"},
        # Rounded to the nearest 5%, because the curve is not precise enough to
        # justify announcing a threshold to the percentage point.
        title=f"Net value peaks around {round(best['threshold'] * 20) * 5:.0f}%",
    )
    fig5.update_traces(line_color=BLUE, line_width=2)
    fig5.add_hline(y=0, line_color="#c8c7c1")
    fig5.add_vline(x=best["threshold"], line_dash="dot", line_color="#52514e",
                   annotation_text="optimum", annotation_position="top left")
    fig5.add_vline(x=threshold, line_dash="dot", line_color=ORANGE,
                   annotation_text="current", annotation_position="top right")
    fig5.update_layout(xaxis_tickformat=".0%")
    right.plotly_chart(fig5, use_container_width=True)
    right.caption(
        "The peak should be treated as a range rather than a point. The curve is "
        "measured on a single 1,000-row holdout and is flat and noisy near the "
        "maximum, so selecting an exact threshold from it would fit the holdout "
        "rather than the population."
    )

    flagged = holdout_score >= threshold
    matrix = pd.DataFrame(
        [
            [int((~flagged & (holdout_y == 0)).sum()), int((flagged & (holdout_y == 0)).sum())],
            [int((~flagged & (holdout_y == 1)).sum()), int((flagged & (holdout_y == 1)).sum())],
        ],
        index=["Actually stayed", "Actually churned"],
        columns=["Not called", "Called"],
    )
    st.write(
        f"**Confusion matrix at a {threshold:.0%} threshold "
        f"({len(holdout_y):,} holdout customers)**"
    )
    st.dataframe(matrix, use_container_width=False)
    st.caption(
        f"The top-right cell is contact cost spent on customers who would have "
        f"stayed; the bottom-left is churners not contacted. At £{call_cost} per "
        f"contact, £{save_value:,} per retained customer and a {save_rate:.0%} "
        f"success rate, net value peaks around "
        f"{round(best['threshold'] * 20) * 5:.0f}%. The optimum moves with those "
        "assumptions."
    )

# ------------------------------------------------------------------- method
with method:
    st.markdown(
        f"""
**Data.** {len(df):,} simulated customers, generated by `make_data.py`.
No real customer data is used anywhere in this demo.

**Model.** Gradient boosting on five features, one-hot encoding for region and
plan, trained in `train_model.py` and saved to `model.pkl`. The app only loads it.

**Performance.** AUC {bundle['auc']:.3f} on a {len(holdout_y):,}-row holdout, against
a tenure-only baseline of {bundle['baseline']:.3f}. Every figure on the decision
threshold tab is measured on that same holdout, never on the rows the model was
trained on.

**Explanations.** The bars on the scoring tab are one-at-a-time counterfactuals:
each field is swapped for the reference customer's value and the customer scored
again. This is straightforward to interpret, but it is not a Shapley value, and
the parts will not sum to the whole where features interact.

**Limitations.** The data is synthetic, so the relationships are cleaner than
anything observed in practice. There is no time-based validation. The cost model
rests on three sidebar assumptions rather than agreed finance figures, and the
value of a retained customer is treated as constant across the base, which it
would not be.
        """
    )
