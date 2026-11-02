"""
PAGE 2: DATA EXPLORER.
Interactive dataset inspection, schema verification, missing value audit, and data quality reporting.
"""

import streamlit as st
import pandas as pd
import json

import config
from src.ui_helpers import render_header, get_data_loader

st.set_page_config(page_title="Data Explorer | Yield Advisor", layout="wide")
render_header(
    title="Data Explorer & Quality Audit",
    subtitle="Inspect Raw & Preprocessed Tables, Verify Schemas, and Audit Quality Gates"
)

loader = get_data_loader()

dataset_options = {
    "Manufacturing Lots (lots.csv)": loader.load_lots,
    "Wafer Summary & Yield (wafer_summary.csv)": loader.load_wafer_summary,
    "Test Program Limits (test_limits.csv)": loader.load_test_limits,
    "Bin Definitions (bin_definitions.csv)": loader.load_bin_definitions,
    "MES Equipment Genealogy (lot_context_mes.csv)": loader.load_lot_context_mes,
    "Fab WAT/PCM E-Test (wat_pcm.csv)": loader.load_wat_pcm,
    "Final Test & Burn-in (final_test_burnin.csv)": loader.load_final_test_burnin,
    "Wafer Sort Die Results (Sample: L262200)": lambda: loader.load_sort_die_results(lot_id="L262200"),
}

selected_name = st.selectbox("📂 Select Dataset to Explore", list(dataset_options.keys()))

with st.spinner(f"Loading {selected_name}..."):
    df = dataset_options[selected_name]()

st.markdown(f"**Dataset Dimensions:** `{df.shape[0]:,}` rows × `{df.shape[1]}` columns")

tab_preview, tab_schema, tab_missing, tab_stats, tab_quality = st.tabs([
    "📋 Data Preview",
    "🧬 Schema & Types",
    "🔍 Missing Values & Duplicates",
    "📊 Statistical Summary",
    "🛡️ Data Quality Report"
])

with tab_preview:
    st.subheader("Data Preview")
    # Quick filter
    if "lot_id" in df.columns:
        all_lots = ["All Lots"] + list(df["lot_id"].unique())
        selected_lot = st.selectbox("Filter by Lot ID", all_lots)
        if selected_lot != "All Lots":
            df_view = df[df["lot_id"] == selected_lot]
        else:
            df_view = df
    else:
        df_view = df

    st.dataframe(df_view.head(200), width="stretch")

with tab_schema:
    st.subheader("Column Schema & Data Types")
    schema_info = []
    for col in df.columns:
        schema_info.append({
            "Column Name": col,
            "Data Type": str(df[col].dtype),
            "Non-Null Count": int(df[col].notna().sum()),
            "Null Count": int(df[col].isna().sum()),
            "Unique Values": int(df[col].nunique()),
            "Sample Value": str(df[col].iloc[0]) if len(df) > 0 else "N/A"
        })
    st.dataframe(pd.DataFrame(schema_info), hide_index=True, width="stretch")

with tab_missing:
    st.subheader("Missing Value & Duplicate Audit")
    col_m1, col_m2 = st.columns(2)
    
    with col_m1:
        st.markdown("#### Missing Values per Column")
        null_counts = df.isna().sum()
        null_pct = (null_counts / len(df)) * 100
        null_df = pd.DataFrame({
            "Column": df.columns,
            "Null_Count": null_counts,
            "Null_Percentage": null_pct.round(2)
        }).sort_values("Null_Count", ascending=False)
        st.dataframe(null_df[null_df["Null_Count"] > 0], hide_index=True, width="stretch")
        if null_counts.sum() == 0:
            st.success("✅ Zero missing values in this dataset.")
        else:
            st.info("ℹ️ In wafer sort records, subsequent tests on failing dies are intentionally NaN due to standard ATE stop-on-fail protocol.")

    with col_m2:
        st.markdown("#### Duplicate Records Check")
        dup_count = df.duplicated().sum()
        if dup_count == 0:
            st.success("✅ No full-row duplicate records detected.")
        else:
            st.warning(f"⚠️ {dup_count} duplicate rows detected.")

with tab_stats:
    st.subheader("Numeric Summary Statistics")
    st.dataframe(df.describe().T.round(3), width="stretch")

with tab_quality:
    st.subheader("Automated Data Quality Audit Report")
    dq_file = config.REPORTS_DIR / "data_quality_report.json"
    if dq_file.exists():
        with open(dq_file, "r") as f:
            dq_rep = json.load(f)
        
        st.json(dq_rep)
        st.download_button(
            label="📥 Download Data Quality Report (JSON)",
            data=json.dumps(dq_rep, indent=2),
            file_name="data_quality_report.json",
            mime="application/json"
        )
    else:
        st.info("Quality report not yet generated. Please execute python train.py.")
