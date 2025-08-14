# ETA-Blame-Lens
# app.py — Streamlit ETA Blame Lens (Shipment-level accuracy + reasons)
# --------------------------------------------------------------
# Quickstart
#   1) Install deps:  pip install -r requirements.txt
#   2) Run:          streamlit run app.py
#   3) Upload your ping-level CSV and map the columns in the sidebar.
#
# What it does
# - Aggregates ping-level ETA predictions to shipment-level accuracy buckets
# - Computes % accuracy within ±30/45/60 (customizable)
# - Attributes top reasons per shipment using selected feature columns
# - Lets you download the shipment-level table as CSV
# --------------------------------------------------------------

from __future__ import annotations
import io
import sys
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="ETA Blame Lens • Streamlit", layout="wide")

# -------------------------
# Utilities
# -------------------------
@dataclass
class ColumnMapping:
    shipment_id: str
    eta_ts: str
    actual_delivery_ts: str
    reason_cols: List[str]


def try_parse_datetime(series: pd.Series) -> pd.Series:
    """Attempt to parse to datetime; leave as-is on failure."""
    if pd.api.types.is_datetime64_any_dtype(series):
        return series
    try:
        return pd.to_datetime(series, errors="coerce")
    except Exception:
        return series


def infer_datetime_cols(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """Suggest ETA and actual columns by name heuristics."""
    lower_cols = {c: c.lower() for c in df.columns}
    eta_candidates = [c for c in df.columns if "eta" in lower_cols[c] and "created" not in lower_cols[c]]
    actual_candidates = [c for c in df.columns if ("actual" in lower_cols[c] and ("time" in lower_cols[c] or "ts" in lower_cols[c])) or "actual_delivery" in lower_cols[c]]
    return eta_candidates, actual_candidates


def compute_buckets(abs_err_min: pd.Series, edges: List[int]) -> pd.Categorical:
    """Bucketize absolute error minutes given edges like [30,45,60]."""
    edges_sorted = sorted(list(set(edges)))
    bins = [-np.inf] + edges_sorted + [np.inf]
    labels = [f"±{edges_sorted[0]} min"]
    for i in range(1, len(edges_sorted)):
        labels.append(f">{edges_sorted[i-1]}–±{edges_sorted[i]} min")
    labels.append(f">{edges_sorted[-1]} min")
    # Replace the 1st label to a friendlier ±X for <= first edge
    return pd.cut(abs_err_min, bins=bins, labels=labels)


def summarize_reasons(df: pd.DataFrame, shipment_col: str, reason_cols: List[str], thresholds: Dict[str, float]) -> pd.Series:
    """Return top 3 reasons per shipment by frequency above threshold."""
    # Build boolean flags
    flags = {}
    for c in reason_cols:
        thr = thresholds.get(c, 0.0)
        # Coerce to numeric where possible
        v = pd.to_numeric(df[c], errors="coerce")
        flags[c] = (v > thr).astype("Int8")
    flag_df = pd.DataFrame(flags)
    groups = flag_df.join(df[[shipment_col]]) .groupby(shipment_col)
    out = {}
    for sid, g in groups:
        sums = g.drop(columns=[shipment_col]).sum(numeric_only=True)
        # Rank by frequency, drop zeros
        top = sums[sums > 0].sort_values(ascending=False).head(3)
        if top.empty:
            out[sid] = ""
        else:
            # format as Reason (count)
            out[sid] = ", ".join([f"{name} ({int(val)})" for name, val in top.items()])
    return pd.Series(out)


def build_shipment_table(df: pd.DataFrame, mapping: ColumnMapping, bucket_edges: List[int]) -> pd.DataFrame:
    # Parse datetimes
    df = df.copy()
    df[mapping.eta_ts] = try_parse_datetime(df[mapping.eta_ts])
    df[mapping.actual_delivery_ts] = try_parse_datetime(df[mapping.actual_delivery_ts])

    # Drop rows with missing required timestamps
    df = df.dropna(subset=[mapping.eta_ts, mapping.actual_delivery_ts, mapping.shipment_id])

    # Error in minutes
    eta_err_min = (df[mapping.eta_ts] - df[mapping.actual_delivery_ts]).dt.total_seconds() / 60.0
    df["abs_error_min"] = eta_err_min.abs()

    # Buckets
    df["error_bucket"] = compute_buckets(df["abs_error_min"], bucket_edges)

    # Shipment-level bucket %
    bucket_counts = df.groupby([mapping.shipment_id, "error_bucket"]).size().unstack(fill_value=0)
    bucket_perc = (bucket_counts.div(bucket_counts.sum(axis=1), axis=0) * 100).round(2)

    # Reason attribution
    thresholds = {}
    for c in mapping.reason_cols:
        thresholds[c] = st.session_state.get(f"thr__{c}", 0.0)

    reasons = summarize_reasons(df, mapping.shipment_id, mapping.reason_cols, thresholds)

    final = bucket_perc.reset_index()
    # Nice column names
    final.columns = [mapping.shipment_id] + [f"acc_{c}" for c in final.columns[1:]]
    final["total_pings"] = df.groupby(mapping.shipment_id).size().reindex(final[mapping.shipment_id]).values
    final["top_reasons"] = final[mapping.shipment_id].map(reasons)
    # Reorder
    cols = [mapping.shipment_id, "total_pings"] + [c for c in final.columns if c.startswith("acc_")] + ["top_reasons"]
    return final[cols]


# -------------------------
# Sidebar — Inputs
# -------------------------
st.sidebar.title("⚙️ Settings")

uploaded = st.sidebar.file_uploader("Upload ping-level CSV", type=["csv"]) 

if uploaded is None:
    st.info("Upload a ping-level CSV to begin. Expect columns like shipment_id, ETA timestamp, actual delivery timestamp, and optional reason columns (e.g., ping_gap, dock_dwell, speed_drop, sequence_miss, appt_slack, detour).")
    st.stop()

# Read CSV
try:
    df = pd.read_csv(uploaded)
except Exception as e:
    st.error(f"Failed to read CSV: {e}")
    st.stop()

st.sidebar.subheader("Column mapping")

# Guess columns
eta_guess, act_guess = infer_datetime_cols(df)
shipment_col = st.sidebar.selectbox("Shipment ID column", options=df.columns.tolist(), index=max(0, df.columns.tolist().index("shipment_id") if "shipment_id" in df.columns else 0))
eta_col = st.sidebar.selectbox("ETA timestamp column", options=df.columns.tolist(), index=(df.columns.tolist().index(eta_guess[0]) if eta_guess else 0))
actual_col = st.sidebar.selectbox("Actual delivery timestamp column", options=df.columns.tolist(), index=(df.columns.tolist().index(act_guess[0]) if act_guess else 0))

# Reason columns
st.sidebar.subheader("Reason attribution")
reason_candidates = [c for c in df.columns if c not in {shipment_col, eta_col, actual_col}]
reason_cols = st.sidebar.multiselect(
    "Select columns to use as reasons",
    options=reason_candidates,
    default=[c for c in ["ping_gap", "dock_dwell", "speed_drop", "sequence_miss", "appt_slack", "detour"] if c in df.columns]
)

# Per-reason thresholds
if reason_cols:
    st.sidebar.caption("Thresholds: a reason counts when column value > threshold")
    thr_cols = st.sidebar.columns(2)
    for i, c in enumerate(reason_cols):
        key = f"thr__{c}"
        with (thr_cols[i % 2]):
            st.session_state[key] = st.number_input(f"{c}", value=0.0, step=1.0, key=key)

# Buckets
st.sidebar.subheader("Accuracy buckets (minutes)")
min30 = st.sidebar.number_input("Bucket 1 edge (±)", min_value=1, value=30, step=5)
min45 = st.sidebar.number_input("Bucket 2 edge (±)", min_value=min30+1, value=45, step=5)
min60 = st.sidebar.number_input("Bucket 3 edge (±)", min_value=min45+1, value=60, step=5)

bucket_edges = [int(min30), int(min45), int(min60)]

mapping = ColumnMapping(
    shipment_id=shipment_col,
    eta_ts=eta_col,
    actual_delivery_ts=actual_col,
    reason_cols=reason_cols,
)

# -------------------------
# Main — Processing & Output
# -------------------------
st.title("ETA Blame Lens — Shipment-level Accuracy & Reasons")

with st.expander("Preview first 10 rows of uploaded data"):
    st.dataframe(df.head(10))

with st.spinner("Computing shipment-level accuracy buckets…"):
    try:
        out = build_shipment_table(df, mapping, bucket_edges)
    except Exception as e:
        st.error(f"Processing failed: {e}")
        st.stop()

st.success("Done")
st.subheader("Shipment-level accuracy (% of pings in each bucket)")
st.dataframe(out, use_container_width=True)

# Download
csv_buf = io.StringIO()
out.to_csv(csv_buf, index=False)
st.download_button(
    label="Download results as CSV",
    data=csv_buf.getvalue().encode("utf-8"),
    file_name="shipment_eta_accuracy_buckets.csv",
    mime="text/csv",
)

# Per-shipment details
st.subheader("Per-shipment details")
selected_sid = st.selectbox("Select a shipment to inspect", options=out[mapping.shipment_id].tolist())

ship_df = df[df[mapping.shipment_id] == selected_sid].copy()
ship_df[mapping.eta_ts] = try_parse_datetime(ship_df[mapping.eta_ts])
ship_df[mapping.actual_delivery_ts] = try_parse_datetime(ship_df[mapping.actual_delivery_ts])
ship_df["eta_error_min"] = (ship_df[mapping.eta_ts] - ship_df[mapping.actual_delivery_ts]).dt.total_seconds() / 60.0
ship_df["abs_error_min"] = ship_df["eta_error_min"].abs()
ship_df["error_bucket"] = compute_buckets(ship_df["abs_error_min"], bucket_edges)

st.caption("Ping-level rows for the selected shipment (with computed error & bucket):")
st.dataframe(ship_df[[mapping.shipment_id, mapping.eta_ts, mapping.actual_delivery_ts, "eta_error_min", "abs_error_min", "error_bucket"] + reason_cols].sort_values(by=mapping.eta_ts), use_container_width=True)

# Lightweight narrative
st.subheader("Summary for selected shipment")
sel_row = out[out[mapping.shipment_id] == selected_sid].iloc[0]
acc_cols = [c for c in out.columns if c.startswith("acc_")]
acc_str = ", ".join([f"{c.replace('acc_','')}: {sel_row[c]}%" for c in acc_cols])
reasons_str = sel_row["top_reasons"] if isinstance(sel_row["top_reasons"], str) else ""
st.write(f"**Accuracy distribution** — {acc_str}")
if reasons_str:
    st.write(f"**Likely reasons** — {reasons_str}")
else:
    st.write("**Likely reasons** — none detected above thresholds.")

# -------------------------
# Footer / Help
# -------------------------
st.divider()
st.caption(
    "Pro tip: Save these settings as defaults by committing your selected column names "
    "and thresholds into the code, or wrap this app in a Docker image for quick deploy."
)

# -------------------------
# requirements.txt (save as a separate file)
# streamlit==1.37.1
# pandas>=2.0.0
# numpy
