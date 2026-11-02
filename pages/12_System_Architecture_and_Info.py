"""
PAGE 12: SYSTEM ARCHITECTURE & SYSTEM INFORMATION.
Technical topology, execution tiers, runtime diagnostics, and semiconductor system specifications.
"""

import streamlit as st
import pandas as pd
import platform
import sys

import config
from src.ui_helpers import render_header

st.set_page_config(page_title="System Architecture & Info | Yield Advisor", layout="wide")
render_header(
    title="System Architecture & Infrastructure Diagnostics",
    subtitle="Deployment Topology, Latency Tiers, Software Environment, and Runtime System Diagnostics"
)

# Top Status Cards
s1, s2, s3, s4 = st.columns(4)
with s1:
    st.metric("System Mode", "Hybrid Tester-Edge / Central", delta="Level 1 Advisory")
with s2:
    st.metric("Product Platform", "AX7-SOC (7nm Node)", delta="FAB2 Cleanroom")
with s3:
    st.metric("Python Runtime", f"v{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
with s4:
    st.metric("Platform Architecture", platform.system(), delta=platform.machine())

st.markdown("---")

tab_topo, tab_latency, tab_env, tab_governance = st.tabs([
    "🏗️ Deployment Topology & Lakehouse",
    "⚡ Latency Budgets & Execution Tiers",
    "💻 Runtime Environment & Diagnostics",
    "🛡️ Human-in-the-Loop & Governance"
])

with tab_topo:
    st.subheader("Hybrid Edge / Central Lakehouse Topology (Technical Design Section 4.1)")
    st.markdown("""
    The AI-Optimized Yield Advisor utilizes a split hybrid architecture engineered for semiconductor manufacturing:
    - **Tester-Cell Edge Agent:** Located on test-floor compute adjacent to Advantest/Teradyne testers.
      Tails STDF files in real-time, executing sub-second streaming outlier checks (DPAT, GDBN, contact alarms) without blocking tester throughput.
    - **Streaming Bus (Kafka / Redpanda):** Durable, partitioned event bus delivering die-level records with exactly-once semantics.
    - **Central Lakehouse (Delta Lake / Iceberg):** Columnar storage with time-travel capabilities for full wafer reproducibility.
    - **Feature Store & Model Serving:** Offline feature engineering and low-latency cache for batch inference.
    - **Advisor & Presentation Layer:** Streamlit cockpit, Plotly interactive visualizations, and offline RAG engineering assistant.
    """)

    st.markdown("""
    ```text
    [ ATE Tester Cell (CP/FT) ]
           │
           ▼
    [ Edge Agent (Rust STDF / ONNX) ] ──(Sub-Second Real-Time Alarms)
           │
           ▼ (Streaming Bus: Kafka)
    [ Central Lakehouse (Iceberg/Parquet) ]
           │
     ┌─────┴──────────────┬──────────────────┬─────────────────┐
     ▼                    ▼                  ▼                 ▼
[Wafer Map AI]    [Excursion SPC]    [Root Cause RCA]   [Yield Predictor]
(Macro F1 >=0.93) (EWMA/CUSUM Alarms)(Permutation Trees) (Pre-Probe Forecast)
     │                    │                  │                 │
     └────────────────────┼──────────────────┴─────────────────┘
                          ▼
            [ Advisory Layer & Inbox ] ──> [ Streamlit Web UI Cockpit ]
                          │
                          ▼
           [ Human Change Control Board (CCB) ]
    ```
    """)

with tab_latency:
    st.subheader("Multi-Tier Latency SLA Specifications (Section 4.2)")
    latency_specs = [
        {"Latency Tier": "Real-Time Edge", "Latency Budget": "< 1 second / die", "Engines & Actions": "Contact resistance drift alarms, consecutive-fail tester halts, real-time NNR", "Target Stage": "Wafer Sort (CP1/CP2)"},
        {"Latency Tier": "Near Real-Time", "Latency Budget": "< 2 minutes / wafer", "Engines & Actions": "Wafer map spatial signature classification, per-wafer DPAT limits, outlier inking", "Target Stage": "Wafer Sort Unload"},
        {"Latency Tier": "Lot End", "Latency Budget": "< 15 minutes / lot", "Engines & Actions": "EWMA/CUSUM excursion detection, lot hold recommendations, retest disposition", "Target Stage": "Lot Disposition Gate"},
        {"Latency Tier": "Analytical Batch", "Latency Budget": "Daily / Retraining", "Engines & Actions": "Root cause attribution, test time optimization pruning, yield model retraining", "Target Stage": "Central Engineering Review"}
    ]
    st.dataframe(pd.DataFrame(latency_specs), hide_index=True, use_container_width=True)

with tab_env:
    st.subheader("Software Stack & Environment Diagnostics")
    env_info = [
        {"Parameter": "Application Name", "Configured Value": "AI-Optimized Yield Advisor"},
        {"Parameter": "Project Version", "Configured Value": "Release 1.0 (Verilumen Labs Technical Evaluation)"},
        {"Parameter": "Root Directory", "Configured Value": str(config.PROJECT_ROOT)},
        {"Parameter": "Data Directory", "Configured Value": str(config.DATA_DIR)},
        {"Parameter": "Models Directory", "Configured Value": str(config.MODELS_DIR)},
        {"Parameter": "Operating System", "Configured Value": f"{platform.system()} {platform.release()}"},
        {"Parameter": "Python Version", "Configured Value": platform.python_version()},
        {"Parameter": "Cross-Platform Path Patch", "Configured Value": "Active (WindowsPath <-> PosixPath auto-bridged)"},
        {"Parameter": "Offline RAG Vector Database", "Configured Value": "Active (Local TF-IDF Vector Space, 19 PDF Pages)"},
        {"Parameter": "Streamlit Framework", "Configured Value": "Multi-Page Native Dashboard"},
    ]
    st.dataframe(pd.DataFrame(env_info), hide_index=True, use_container_width=True)

with tab_governance:
    st.subheader("Human-in-the-Loop & Quality Governance Policy")
    st.markdown("""
    Following semiconductor industry standards (AEC-Q001, IATF 16949, PPAP):
    1. **Advisory Authority (Level 1):** The Yield Advisor operates strictly in advisory mode. No production ATE test limits, test flows, or fab equipment recipes are modified autonomously.
    2. **Evidence Pack Requirement:** Every candidate recommendation requires a quantified economic benefit ($USD), an explicit quality escape risk bound (DPPM), a statistical confidence metric, and traceable raw evidence.
    3. **Change Control Approval (CCB):** Limit changes, test removals, and equipment inhibits generate standard Engineering Change Requests (ECR) for human sign-off.
    4. **Audit Trail:** Every human approval, rejection, and review note is cryptographically logged with timestamps for automotive traceability.
    """)
