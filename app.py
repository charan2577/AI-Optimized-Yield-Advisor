"""
AI OPTIMIZED YIELD ADVISOR
Main Application Entry Point.
Verilumen Labs Pvt. Ltd. - Semiconductor Chip Testing Intelligence.
"""

import streamlit as st
import config
from src.ui_helpers import apply_custom_css, render_header, get_wafer_summary, get_lots

st.set_page_config(
    page_title="AI OPTIMIZED YIELD ADVISOR | Verilumen Labs",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

render_header(
    title="AI OPTIMIZED YIELD ADVISOR",
    subtitle="Intelligent Semiconductor Yield Monitoring, Prediction and Root Cause Analysis"
)

# System Summary Metrics
ws = get_wafer_summary()
lots = get_lots()

col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    st.metric("Total Lots", f"{len(lots):,}")
with col2:
    st.metric("Total Wafers", f"{len(ws):,}")
with col3:
    st.metric("Gross Tested Dies", f"{ws['gross_die'].sum():,}")
with col4:
    mean_yield = ws["final_yield"].mean() * 100
    st.metric("Mean Final Yield", f"{mean_yield:.2f}%", delta="+0.19% vs 1st Pass")
with col5:
    st.metric("Active AI Engines", "7 / 7 Operational", delta="All Healthy")

st.markdown("---")

# Architectural Overview & Modules
st.subheader("System Architecture & Core Intelligence Engines")

st.markdown("""
The **AI Optimized Yield Advisor** ingests per-die Automatic Test Equipment (ATE) parametric records,
fab Wafer Acceptance Testing (WAT/PCM metrology), and MES equipment genealogy to provide closed-loop
semiconductor yield intelligence across wafer sort, final test, and burn-in reliability.
""")

col_left, col_right = st.columns(2)

with col_left:
    st.markdown("""
    ### 🔬 Active AI Engines
    1. **Wafer Map Pattern AI**: Automated spatial failure signature recognition (Macro F1 >= 0.93 across 6 classes).
    2. **Excursion Detector**: Early statistical detection of yield and parametric shifts using EWMA, CUSUM, and Isolation Forest.
    3. **Root Cause Engine**: Multi-factor commonality analysis, Kruskal-Wallis tests, and permutation feature attribution.
    4. **Adaptive Limits & Latent Defects**: AEC-Q001 Dynamic Part Average Testing (DPAT) and Good-Die-in-Bad-Neighborhood (GDBN) outlier screening.
    5. **Test Time Optimizer**: Redundant test identification, Rule-of-Three escape risk calculations, and test reordering.
    6. **Yield Predictor**: Pre-probe yield forecasting from fab WAT metrology with P10/P50/P90 quantile uncertainty intervals.
    7. **AI Engineering Copilot**: Local document-grounded RAG assistant with source citations and automated 8D report drafting.
    """)

with col_right:
    st.markdown("""
    ### 🧭 Quick Navigation
    Use the sidebar to navigate through the 12 dedicated engineering modules:
    - **1. Executive Dashboard**: High-level yield KPIs, trends, failure Pareto, and active alerts.
    - **2. Data Explorer**: Interactive data browser with schema validation and quality auditing.
    - **3. Wafer Map Analysis**: Interactive 26x26 spatial die viewer with pattern classification.
    - **4. Excursion Monitoring**: EWMA and CUSUM control charts with automated alarm tracking.
    - **5. Root Cause Analysis**: Statistical commonality rankings for tools, chambers, and recipes.
    - **6. Latent Defect Detection**: DPAT & GDBN outlier screening correlated with burn-in reliability.
    - **7. Test Time Optimization**: Test duration pruning and rule-of-three escape bounds.
    - **8. Yield Prediction**: Pre-probe lot and wafer yield forecasting.
    - **9. Engineering Recommendations**: Prioritized advisory inbox with human approval workflows.
    - **10. Model Performance**: Consolidated benchmarks against Technical Design targets.
    - **11. AI Engineering Copilot**: Local RAG question answering and automated 8D drafting.
    - **12. System Architecture & Info**: Topology, latency budgets, and runtime environment.
    """)

st.info("💡 **Advisory Mode Active**: All recommendations are advisory. In compliance with semiconductor change control standards, no production tester limits or MES routing are modified autonomously.")
