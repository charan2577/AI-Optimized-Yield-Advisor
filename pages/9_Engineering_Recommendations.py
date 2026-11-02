"""
PAGE 9: ENGINEERING RECOMMENDATIONS.
Prioritized Advisory Inbox, Evidence Packs, Benefit & Risk Quantification,
and Human-in-the-Loop Governance Workflow.
"""

import streamlit as st
import pandas as pd

import config
from src.ui_helpers import render_header, get_recommendation_engine

st.set_page_config(page_title="Engineering Recommendations | Yield Advisor", layout="wide")
render_header(
    title="Engineering Recommendations & Decision Inbox",
    subtitle="Risk-Adjusted Decision Support, Traceable Evidence Packs, and Change Control Approval Governance"
)

rec_engine = get_recommendation_engine()
recs = rec_engine.recommendations

# Filter controls
col_f1, col_f2, col_f3 = st.columns([2, 2, 2])
with col_f1:
    filter_priority = st.multiselect("Filter Priority", ["P1", "P2", "P3"], default=["P1", "P2", "P3"])
with col_f2:
    filter_status = st.multiselect("Filter Status", ["Pending Review", "Approved", "Rejected", "Completed"], default=["Pending Review", "Approved"])
with col_f3:
    st.info("🔒 **Governance:** Autonomous parameter change is prohibited. Human review required for Level 1 advice.")

st.markdown("---")

filtered_recs = [
    r for r in recs 
    if r.get("priority") in filter_priority and r.get("status") in filter_status
]

if not filtered_recs:
    st.info("No recommendations match the selected filters.")
else:
    for idx, r in enumerate(filtered_recs):
        sev = r.get("severity", "MEDIUM")
        badge = "badge-critical" if sev == "CRITICAL" else ("badge-high" if sev == "HIGH" else "badge-medium")
        prio = r.get("priority", "P2")

        with st.container():
            st.markdown(
                f"""
                <div style="border-left: 6px solid #2563eb; background: #ffffff; padding: 18px; margin-bottom: 16px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.08);">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <h3 style="margin: 0; color: #0f172a;">{r['id']}: {r['title']}</h3>
                        <div>
                            <span style="background: #1e293b; color: white; padding: 4px 10px; border-radius: 4px; font-weight: 700; font-size: 0.8rem; margin-right: 6px;">{prio}</span>
                            <span class="{badge}">{sev}</span>
                            <span style="margin-left: 8px; font-weight: 600; color: #475569; font-size: 0.85rem;">Status: <b>{r['status']}</b></span>
                        </div>
                    </div>
                    <div style="margin: 8px 0; color: #64748b; font-size: 0.88rem;">
                        <b>Engine:</b> {r['engine_source']} | <b>Scope:</b> {r['affected_scope']} | <b>Assigned Owner:</b> {r.get('owner', 'Engineering')}
                    </div>
                    <p style="margin: 8px 0; color: #334155; font-size: 0.95rem;">
                        <b>Supporting Evidence:</b> {r['evidence']}
                    </p>
                    <div style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 12px; background: #f8fafc; padding: 12px; border-radius: 6px; margin: 10px 0; font-size: 0.88rem;">
                        <div><b>Model Confidence:</b> <span style="color: #2563eb; font-weight: 700;">{r.get('confidence_pct', 90):.1f}%</span></div>
                        <div><b>Expected Benefit:</b> <span style="color: #10b981; font-weight: 700;">${r.get('expected_benefit_usd', 0):,.2f}/yr</span></div>
                        <div><b>Quality Risk:</b> <span style="color: #64748b;">{r.get('quality_risk', 'None')}</span></div>
                    </div>
                    <div style="background: #eff6ff; border: 1px solid #bfdbfe; padding: 10px; border-radius: 6px; font-size: 0.9rem; color: #1e40af; margin-bottom: 12px;">
                        <b>Recommended Action:</b> {r['recommended_action']}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

            # Interactive Decision Controls
            col_act1, col_act2, col_act3, col_notes = st.columns([1.2, 1.2, 1.2, 4])
            with col_act1:
                if st.button("✅ Approve", key=f"app_{r['id']}"):
                    rec_engine.update_status(r["id"], "Approved")
                    st.rerun()
            with col_act2:
                if st.button("❌ Reject", key=f"rej_{r['id']}"):
                    rec_engine.update_status(r["id"], "Rejected")
                    st.rerun()
            with col_act3:
                if st.button("🏁 Mark Done", key=f"done_{r['id']}"):
                    rec_engine.update_status(r["id"], "Completed")
                    st.rerun()
            with col_notes:
                notes = st.text_input("Engineer Review Notes", key=f"note_{r['id']}", placeholder="Enter CCB ECR ticket number or action comments...")
                if notes and st.button("Save Note", key=f"savenote_{r['id']}"):
                    rec_engine.update_status(r["id"], r["status"], engineer_notes=notes)
                    st.success("Note recorded.")
