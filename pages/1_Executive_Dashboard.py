"""
PAGE 1: EXECUTIVE DASHBOARD.
High-level semiconductor yield KPIs, lot yield trends, failure Pareto, and active alerts.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

import config
from src.ui_helpers import (
    render_header,
    get_wafer_summary,
    get_lots,
    get_data_loader,
    get_excursion_detector,
    get_recommendation_engine,
)

st.set_page_config(page_title="Executive Dashboard | Yield Advisor", layout="wide")
render_header(
    title="Executive Yield Cockpit",
    subtitle="Enterprise KPI Monitoring, Failure Pareto, and Excursion Alerts"
)

loader = get_data_loader()
ws = get_wafer_summary()
lots = get_lots()
detector = get_excursion_detector()
rec_engine = get_recommendation_engine()
recs = rec_engine.recommendations

# Compute Top KPIs
total_lots = len(lots)
total_wafers = len(ws)
gross_dies = int(ws["gross_die"].sum())
first_pass_yield = float(ws["first_pass_yield"].mean() * 100)
final_yield = float(ws["final_yield"].mean() * 100)
yield_recovery = final_yield - first_pass_yield

# Burn-in fails
stats_json = loader.load_dataset_stats()
burnin_fails = stats_json.get("burnin_fails", 700)

# Detected excursions
lot_charts = detector.compute_lot_yield_charts(ws, lots) if detector else pd.DataFrame()
excursion_count = int(lot_charts["is_excursion_lot"].sum()) if not lot_charts.empty else 0
p1_recs = len([r for r in recs if r.get("priority") == "P1"])

# Render KPI Grid
kpi_cols = st.columns(5)
with kpi_cols[0]:
    st.metric("Total Tested Lots", f"{total_lots}")
    st.metric("Total Wafers", f"{total_wafers:,}")
with kpi_cols[1]:
    st.metric("Gross Tested Dies", f"{gross_dies:,}")
    st.metric("Burn-in Failures", f"{burnin_fails:,}", delta="Reliability Risk", delta_color="inverse")
with kpi_cols[2]:
    st.metric("First-Pass Yield", f"{first_pass_yield:.2f}%")
    st.metric("Final Sort Yield", f"{final_yield:.2f}%", delta=f"+{yield_recovery:.2f}% recovered")
with kpi_cols[3]:
    st.metric("Detected Excursions", f"{excursion_count} Lots", delta="EWMA/CUSUM Alarms", delta_color="inverse")
    st.metric("Overkill Recovery", f"{yield_recovery:.2f} pts", delta="From Retest")
with kpi_cols[4]:
    st.metric("High-Priority Alerts", f"{p1_recs} P1 Actions", delta="Pending Review", delta_color="inverse")
    st.metric("Est. Annual Impact", "$1.85M", delta="Yield + Test Time")

st.markdown("---")

# Row 1: Lot Yield Trend & Wafer Yield Distribution
col_trend, col_dist = st.columns([3, 2])

with col_trend:
    st.subheader("📈 Chronological Lot Yield Trend")
    if not lot_charts.empty:
        fig_trend = go.Figure()
        fig_trend.add_trace(go.Scatter(
            x=lot_charts["lot_id"],
            y=lot_charts["lot_yield"] * 100,
            mode="lines+markers",
            name="Actual Lot Yield (%)",
            line=dict(color="#2563eb", width=2),
            marker=dict(size=7)
        ))
        # Baseline target line
        mu0 = float(lot_charts["baseline_mu0"].iloc[0] * 100)
        fig_trend.add_trace(go.Scatter(
            x=lot_charts["lot_id"],
            y=[mu0] * len(lot_charts),
            mode="lines",
            name="Baseline Target (91.7%)",
            line=dict(color="#10b981", dash="dash", width=2)
        ))
        # Excursion points
        ex_lots = lot_charts[lot_charts["is_excursion_lot"]]
        if not ex_lots.empty:
            fig_trend.add_trace(go.Scatter(
                x=ex_lots["lot_id"],
                y=ex_lots["lot_yield"] * 100,
                mode="markers",
                name="Excursion Alarm",
                marker=dict(color="#ef4444", size=10, symbol="x")
            ))
        fig_trend.update_layout(
            height=360,
            margin=dict(l=20, r=20, t=30, b=20),
            xaxis_title="Lot Identifier (Chronological Order)",
            yaxis_title="Sort Yield (%)",
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_trend, width="stretch")

with col_dist:
    st.subheader("📊 Wafer Yield Distribution")
    fig_hist = px.histogram(
        ws,
        x=ws["final_yield"] * 100,
        nbins=30,
        color_discrete_sequence=["#3b82f6"],
        labels={"x": "Final Wafer Yield (%)"}
    )
    fig_hist.add_vline(x=final_yield, line_dash="dash", line_color="#ef4444", annotation_text=f"Mean: {final_yield:.2f}%")
    fig_hist.update_layout(
        height=360,
        margin=dict(l=20, r=20, t=30, b=20),
        xaxis_title="Wafer Yield (%)",
        yaxis_title="Wafer Count",
        showlegend=False
    )
    st.plotly_chart(fig_hist, width="stretch")

# Row 2: Soft-Bin Failure Pareto & Yield by Lot
col_pareto, col_lots = st.columns([3, 2])

with col_pareto:
    st.subheader("🎯 Soft-Bin Failure Pareto Chart")
    pareto_data = stats_json.get("fail_pareto_final", {})
    if pareto_data:
        p_df = pd.DataFrame(list(pareto_data.items()), columns=["Soft_Bin", "Fail_Count"]).sort_values("Fail_Count", ascending=False)
        p_df["Cumulative_Pct"] = (p_df["Fail_Count"].cumsum() / p_df["Fail_Count"].sum()) * 100

        fig_p = go.Figure()
        fig_p.add_trace(go.Bar(
            x=p_df["Soft_Bin"],
            y=p_df["Fail_Count"],
            name="Fail Count",
            marker_color="#475569"
        ))
        fig_p.add_trace(go.Scatter(
            x=p_df["Soft_Bin"],
            y=p_df["Cumulative_Pct"],
            name="Cumulative %",
            yaxis="y2",
            line=dict(color="#f97316", width=2),
            marker=dict(size=6)
        ))
        fig_p.update_layout(
            height=360,
            margin=dict(l=20, r=20, t=30, b=20),
            yaxis=dict(title="Failing Die Count"),
            yaxis2=dict(title="Cumulative %", overlaying="y", side="right", range=[0, 105]),
            hovermode="x",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_p, width="stretch")

with col_lots:
    st.subheader("🚨 Recent High-Priority Alerts")
    if recs:
        p1_list = [r for r in recs if r.get("priority") in ["P1", "P2"]][:5]
        alert_rows = []
        for r in p1_list:
            alert_rows.append({
                "Priority": r["priority"],
                "Title": r["title"],
                "Scope": r["affected_scope"],
                "Severity": r["severity"],
                "Status": r["status"]
            })
        st.dataframe(pd.DataFrame(alert_rows), hide_index=True, width="stretch")
    else:
        st.info("No active high-priority alerts.")
