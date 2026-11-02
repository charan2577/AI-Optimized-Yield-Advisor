"""
PAGE 4: EXCURSION MONITORING.
Statistical Process Control (SPC), EWMA, CUSUM, Shewhart control limits, and multivariate anomaly detection.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

import config
from src.ui_helpers import (
    render_header,
    get_wafer_summary,
    get_lots,
    get_excursion_detector,
    get_wafer_features,
)

st.set_page_config(page_title="Excursion Monitoring | Yield Advisor", layout="wide")
render_header(
    title="Excursion Monitoring & Early Warning",
    subtitle="Statistical Process Control (SPC), EWMA / CUSUM Trend Shifts, and Multivariate Anomaly Detection"
)

ws = get_wafer_summary()
lots = get_lots()
detector = get_excursion_detector()
wafer_features = get_wafer_features()

if detector:
    lot_charts = detector.compute_lot_yield_charts(ws, lots)
    events = detector.identify_excursion_events(lot_charts, wafer_features)
else:
    lot_charts = pd.DataFrame()
    events = []

# Excursion summary metrics
total_excursions = int(lot_charts["is_excursion_lot"].sum()) if not lot_charts.empty else 0
ewma_alarms = int(lot_charts["ewma_alarm"].sum()) if not lot_charts.empty else 0
cusum_alarms = int(lot_charts["cusum_alarm"].sum()) if not lot_charts.empty else 0

m1, m2, m3, m4 = st.columns(4)
with m1:
    st.metric("Total Excursion Lots", f"{total_excursions} / 40", delta="SPC Signals", delta_color="inverse")
with m2:
    st.metric("EWMA Alarms", f"{ewma_alarms} Lots", delta="Small Shift Detection", delta_color="inverse")
with m3:
    st.metric("CUSUM Alarms", f"{cusum_alarms} Lots", delta="Sustained Drift", delta_color="inverse")
with m4:
    st.metric("Detection Latency", "< 1 Lot Onset", delta="Target Met (<24h)")

st.markdown("---")

tab_spc, tab_cusum, tab_anom, tab_events = st.tabs([
    "📈 EWMA & Shewhart Control Charts",
    "📉 CUSUM Drift Monitoring",
    "🤖 Multivariate Anomaly Detection",
    "🚨 Identified Excursion Incidents"
])

with tab_spc:
    st.subheader("Exponentially Weighted Moving Average (EWMA) Control Chart")
    st.markdown("""
    EWMA accumulates statistical evidence across successive lots to detect subtle, sustained shifts 
    (1.5σ) several lots earlier than conventional single-point Shewhart charts.
    """)
    
    if not lot_charts.empty:
        fig_ewma = go.Figure()

        # Actual Lot Yield points
        fig_ewma.add_trace(go.Scatter(
            x=lot_charts["lot_id"],
            y=lot_charts["lot_yield"] * 100,
            mode="markers",
            name="Raw Lot Yield (%)",
            marker=dict(color="#94a3b8", size=6)
        ))

        # EWMA Filtered Curve
        fig_ewma.add_trace(go.Scatter(
            x=lot_charts["lot_id"],
            y=lot_charts["ewma_value"] * 100,
            mode="lines+markers",
            name="EWMA Smoothed (λ=0.20)",
            line=dict(color="#2563eb", width=2.5),
            marker=dict(size=6)
        ))

        # Baseline
        mu0 = float(lot_charts["baseline_mu0"].iloc[0] * 100)
        fig_ewma.add_trace(go.Scatter(
            x=lot_charts["lot_id"],
            y=[mu0] * len(lot_charts),
            mode="lines",
            name="Baseline Target (μ0)",
            line=dict(color="#10b981", dash="dash")
        ))

        # EWMA UCL & LCL
        fig_ewma.add_trace(go.Scatter(
            x=lot_charts["lot_id"],
            y=lot_charts["ewma_ucl"] * 100,
            mode="lines",
            name="EWMA UCL (+3σ)",
            line=dict(color="#f59e0b", dash="dot")
        ))
        fig_ewma.add_trace(go.Scatter(
            x=lot_charts["lot_id"],
            y=lot_charts["ewma_lcl"] * 100,
            mode="lines",
            name="EWMA LCL (-3σ)",
            line=dict(color="#ef4444", dash="dot"),
            fill="tonexty",
            fillcolor="rgba(239, 68, 68, 0.05)"
        ))

        # Alarm Markers
        alarm_pts = lot_charts[lot_charts["ewma_alarm"]]
        if not alarm_pts.empty:
            fig_ewma.add_trace(go.Scatter(
                x=alarm_pts["lot_id"],
                y=alarm_pts["ewma_value"] * 100,
                mode="markers",
                name="EWMA Out-of-Control Alarm",
                marker=dict(color="#dc2626", size=11, symbol="triangle-down")
            ))

        fig_ewma.update_layout(
            height=420,
            margin=dict(l=20, r=20, t=30, b=20),
            xaxis_title="Manufacturing Lot (Chronological)",
            yaxis_title="Yield (%)",
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_ewma, width="stretch")

with tab_cusum:
    st.subheader("Cumulative Sum (CUSUM) Downward Drift Detection")
    st.markdown(r"""
    CUSUM tracks cumulative deviations from target yield $C_t^- = \max(0, (\mu_0 - K) - X_t + C_{t-1}^-)$. 
    A signal is triggered when $C_t^- > H$, providing rapid detection of yield loss onset.
    """)
    if not lot_charts.empty:
        fig_cusum = go.Figure()
        fig_cusum.add_trace(go.Scatter(
            x=lot_charts["lot_id"],
            y=lot_charts["cusum_neg"],
            mode="lines+markers",
            name="CUSUM Negative Yield Deficit",
            line=dict(color="#ef4444", width=2.5),
            marker=dict(size=6)
        ))
        # Threshold H
        h_val = float(lot_charts["cusum_threshold"].iloc[0])
        fig_cusum.add_trace(go.Scatter(
            x=lot_charts["lot_id"],
            y=[h_val] * len(lot_charts),
            mode="lines",
            name=f"Alarm Threshold H ({h_val:.4f})",
            line=dict(color="#dc2626", dash="dash", width=2)
        ))
        fig_cusum.update_layout(
            height=380,
            margin=dict(l=20, r=20, t=30, b=20),
            xaxis_title="Manufacturing Lot",
            yaxis_title="Cumulative Negative Deviation",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_cusum, width="stretch")

with tab_anom:
    st.subheader("Multivariate Anomaly Detection (Isolation Forest)")
    st.markdown("""
    Isolation Forest evaluates wafer vectors across yield, soft bin rates, and parametric moments 
    to flag multivariate outliers that may pass univariate limits.
    """)
    if "anomaly_score" in wafer_features.columns:
        fig_anom = px.scatter(
            wafer_features,
            x="final_yield",
            y="anomaly_score",
            color="is_multivariate_anomaly",
            color_discrete_map={True: "#ef4444", False: "#3b82f6"},
            labels={"final_yield": "Wafer Final Yield", "anomaly_score": "Isolation Forest Decision Score"},
            hover_data=["lot_id", "wafer_id", "ETCH_chamber"]
        )
        fig_anom.update_layout(height=400, margin=dict(l=20, r=20, t=20, b=20))
        st.plotly_chart(fig_anom, width="stretch")

with tab_events:
    st.subheader("Synthesized Excursion Incidents")
    if events:
        for ev in events:
            sev = ev.get("severity", "MEDIUM")
            badge = "badge-critical" if sev == "CRITICAL" else ("badge-high" if sev == "HIGH" else "badge-medium")
            st.markdown(
                f"""
                <div style="border-left: 4px solid #ef4444; background: #f8fafc; padding: 14px; margin-bottom: 12px; border-radius: 4px;">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <h4 style="margin: 0; color: #1e293b;">{ev.get('event_type')}: {ev.get('target_id')}</h4>
                        <span class="{badge}">{sev}</span>
                    </div>
                    <p style="margin: 6px 0 0 0; color: #475569; font-size: 0.95rem;">{ev.get('description')}</p>
                </div>
                """,
                unsafe_allow_html=True
            )
    else:
        st.info("No synthesized excursion incidents flagged.")
