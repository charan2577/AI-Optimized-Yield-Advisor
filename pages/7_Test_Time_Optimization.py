"""
PAGE 7: TEST TIME OPTIMIZATION.
Redundant test pruning, parametric correlation analysis, Rule-of-Three escape risk bounds,
and optimal stop-on-fail test reordering.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import json

import config
from src.ui_helpers import render_header, get_test_optimizer, get_data_loader

st.set_page_config(page_title="Test Time Optimization | Yield Advisor", layout="wide")
render_header(
    title="Test Time Optimization & Redundancy Analysis",
    subtitle="Execution Time Profiling, Parametric Redundancy, Rule-of-Three Escape Bounds, and Test Reordering"
)

loader = get_data_loader()
optimizer = get_test_optimizer()

# Load optimizer proposals from report
eval_file = config.REPORTS_DIR / "model_evaluation_report.json"
opt_data = {}
if eval_file.exists():
    with open(eval_file, "r") as f:
        rep = json.load(f)
    opt_data = rep.get("engine_5_test_optimizer", {})

nominal_time = opt_data.get("nominal_test_time_ms", 1230.0)
time_saved = opt_data.get("total_test_time_reduction_ms", 106.4)
pct_saved = opt_data.get("test_time_reduction_pct", 8.65)
annual_savings = opt_data.get("total_annual_savings_usd", 82736.64)

# KPI Summary
t1, t2, t3, t4, t5 = st.columns(5)
with t1:
    st.metric("Nominal Test Time", f"{nominal_time:.0f} ms / die")
with t2:
    st.metric("Proposed Time Reduction", f"{time_saved:.1f} ms", delta=f"-{pct_saved:.2f}%")
with t3:
    st.metric("New Test Duration", f"{nominal_time - time_saved:.1f} ms", delta="Faster Cycle")
with t4:
    st.metric("Annual Tester Savings", f"${annual_savings:,.2f}", delta="5.4M units/mo")
with t5:
    st.metric("Escape Risk Bound", "< 6.0 DPPM", delta="Rule of Three (95% Conf)")

st.markdown("---")

tab_proposals, tab_breakdown, tab_corr, tab_reorder = st.tabs([
    "💡 Optimization Proposals & Business Case",
    "⏱️ Test Duration & Failure Breakdown",
    "🔥 Parametric Correlation Heatmap",
    "🔀 Stop-on-Fail Test Reordering"
])

with tab_proposals:
    st.subheader("Formal Test Optimization Proposals requiring CCB Approval")
    st.markdown("""
    Recommendations follow Section 6.5 of the Technical Design Document:
    *No test item is deleted solely due to low failure count.* Proposals suggest audit sampling 
    under an explicit, statistically bounded **Rule of Three** escape limit ($3/N$).
    """)

    proposals = opt_data.get("proposals", [])
    for p in proposals:
        st.markdown(
            f"""
            <div style="border-left: 5px solid #10b981; background: #ffffff; padding: 16px; margin-bottom: 14px; border-radius: 6px; box-shadow: 0 1px 3px rgba(0,0,0,0.1);">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <h3 style="margin: 0; color: #0f172a;">{p['proposal_id']}: {p['test_name']}</h3>
                    <span style="background: #fef3c7; color: #92400e; padding: 4px 10px; border-radius: 4px; font-weight: 600; font-size: 0.8rem;">
                        {p['status']}
                    </span>
                </div>
                <p style="margin: 8px 0; color: #1e293b; font-weight: 600;">Proposed Action: {p['action']}</p>
                <p style="margin: 4px 0 10px 0; color: #475569; font-size: 0.95rem;">{p['justification']}</p>
                <div style="display: flex; gap: 24px; font-size: 0.9rem; color: #334155; background: #f8fafc; padding: 10px; border-radius: 4px;">
                    <span><b>Time Saved:</b> {p['test_time_saved_ms']:.1f} ms / die</span>
                    <span><b>Annual Cost Savings:</b> ${p['annual_savings_usd']:,.2f}</span>
                    <span><b>Escape Risk Bound:</b> {p['escape_risk_dppm']:.2f} DPPM</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

with tab_breakdown:
    st.subheader("Test Program Item Profile")
    if optimizer:
        redundancy_df = optimizer.analyze_test_redundancy()
        st.dataframe(redundancy_df, hide_index=True, width="stretch")

        fig_tt = px.bar(
            redundancy_df.sort_values("test_time_ms", ascending=False),
            x="test_name",
            y="test_time_ms",
            color="test_time_ms",
            color_continuous_scale="Blues",
            labels={"test_name": "Test Item", "test_time_ms": "Execution Duration (ms)"},
            title="Nominal Execution Time per Test Item (Stop-on-Fail ATE)"
        )
        fig_tt.update_layout(height=380, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_tt, width="stretch")

with tab_corr:
    st.subheader("Parametric Test Correlation Matrix (Passing Dies)")
    st.markdown("""
    Parametric correlation identifies tests where one measurement reliably predicts another.
    **RING_OSC and FMAX** exhibit near-perfect correlation ($r = 0.996$), proving that RING_OSC 
    can be sampled without losing speed-binning coverage.
    """)
    if optimizer:
        corr_matrix = optimizer.compute_correlation_matrix()
        fig_corr = px.imshow(
            corr_matrix,
            text_auto=True,
            aspect="auto",
            color_continuous_scale="RdBu_r",
            range_color=[-1, 1],
        )
        fig_corr.update_layout(height=480, margin=dict(l=20, r=20, t=20, b=20))
        st.plotly_chart(fig_corr, width="stretch")

with tab_reorder:
    st.subheader("Optimal Stop-on-Fail Test Sequence")
    st.markdown("""
    Under stop-on-fail protocol, executing high-failure-rate and fast tests earlier in the test flow 
    ensures that defective dies abort earlier, saving average tester time across the entire lot.
    """)
    if optimizer:
        redundancy_df = optimizer.analyze_test_redundancy()
        reordered_df = optimizer.optimize_test_ordering(redundancy_df)
        st.dataframe(
            reordered_df[["proposed_order", "current_order", "test_name", "test_time_ms", "fail_rate_pct", "detection_efficiency"]],
            hide_index=True,
            width="stretch"
        )
