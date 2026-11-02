"""
PAGE 11: AI ENGINEERING COPILOT.
Offline document-grounded retrieval-augmented generation (RAG) assistant,
specification Q&A, and automated semiconductor 8D problem-solving report drafting.
"""

import streamlit as st
import pandas as pd

import config
from src.ui_helpers import render_header, get_copilot_engine, get_excursion_detector, get_root_cause_engine, get_wafer_features, get_lots, get_wafer_summary

st.set_page_config(page_title="AI Copilot | Yield Advisor", layout="wide")
render_header(
    title="AI Engineering Copilot & 8D Assistant",
    subtitle="Offline Document-Grounded Engineering Retrieval (RAG), Specification Q&A, and Automated 8D Drafting"
)

copilot = get_copilot_engine()
detector = get_excursion_detector()
rc_engine = get_root_cause_engine()
wafer_features = get_wafer_features()
lots = get_lots()
ws = get_wafer_summary()

# Top KPI Status Cards
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.metric("RAG Engine Status", "Active & Local", delta="Zero Paid APIs")
with c2:
    st.metric("Indexed Documents", "Technical Design Rev 1.0", delta="19 PDF Pages + Repos")
with c3:
    st.metric("Semantic Index", f"{len(copilot.chunks) if copilot else 0} Chunks", delta="TF-IDF Dense Vector Space")
with c4:
    st.metric("Governance Mode", "Grounded Only", delta="Hallucination Guardrail")

st.markdown("---")

tab_chat, tab_8d, tab_docs = st.tabs([
    "💬 Engineering Copilot Q&A",
    "📝 Automated 8D Problem Solving Report",
    "📚 Indexed Knowledge Base & Architecture"
])

with tab_chat:
    st.subheader("Semiconductor Yield Engineering Assistant")
    st.markdown("""
    Ask technical questions regarding test procedures, wafer patterns, control limits, or excursion root causes.
    All answers are **100% locally computed** and strictly grounded in approved Verilumen engineering documentation.
    """)

    # Quick prompt buttons
    st.markdown("##### Quick Technical Prompts:")
    quick_col1, quick_col2, quick_col3 = st.columns(3)
    preset_query = ""
    with quick_col1:
        if st.button("🔍 Explain ETCH-07 Chamber C Excursion", use_container_width=True):
            preset_query = "What caused the Edge-Ring yield excursion on ETCH-07 Chamber C?"
        if st.button("📐 How are AEC-Q001 DPAT limits computed?", use_container_width=True):
            preset_query = "How are AEC-Q001 Dynamic Part Average Testing (DPAT) limits calculated?"
    with quick_col2:
        if st.button("📈 EWMA vs Shewhart Detection Latency", use_container_width=True):
            preset_query = "Explain the mathematical difference between EWMA and Shewhart control charts."
        if st.button("⏱️ Rule-of-Three Test Time Reduction", use_container_width=True):
            preset_query = "What is the Rule of Three escape risk bound for test time optimization?"
    with quick_col3:
        if st.button("🔬 Wafer Map Pattern Classes & Mechanisms", use_container_width=True):
            preset_query = "What are the wafer map spatial failure patterns and their equipment causes?"
        if st.button("⚡ Pre-Probe Yield Prediction Architecture", use_container_width=True):
            preset_query = "How does the pre-probe yield prediction engine forecast lot and wafer yield?"

    query_input = st.text_input(
        "Enter your engineering query:",
        value=preset_query,
        placeholder="e.g., Why did lot L262234 drop in yield, or how does GDBN calculate neighbor failures?"
    )

    if query_input:
        with st.spinner("Searching local technical specifications and engineering knowledge base..."):
            response = copilot.query(query_input)

        st.markdown(
            f"""
            <div style="background: #ffffff; border: 1px solid #e2e8f0; border-left: 5px solid #2563eb; padding: 18px; border-radius: 8px; margin-top: 16px; box-shadow: 0 1px 3px rgba(0,0,0,0.06);">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
                    <span style="font-weight: 700; color: #1e293b; font-size: 1.05rem;">Engineering Response</span>
                    <span style="background: {'#dcfce7' if response['confidence']=='High' else '#fef3c7'}; color: {'#166534' if response['confidence']=='High' else '#92400e'}; padding: 4px 10px; border-radius: 4px; font-weight: 600; font-size: 0.8rem;">
                        Confidence: {response['confidence']}
                    </span>
                </div>
                <div style="color: #334155; line-height: 1.6; font-size: 0.95rem;">
                    {response['answer']}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown("#### 📖 Grounded Source Citations:")
        if response.get("sources"):
            for s in response["sources"]:
                with st.expander(f"📄 {s['document']} - Page {s['page']} ({s['section']}) [Relevance: {s['relevance']}]"):
                    st.markdown(f"**Document Path:** `{s['source']}`")
                    st.markdown(f"**Extracted Text Section:**\n> {s['excerpt']}")
        else:
            st.info("No specific documentation excerpts matched the confidence threshold.")

with tab_8d:
    st.subheader("Automated 8D Problem Solving Report Generator")
    st.markdown("""
    Automatically draft formal automotive/industrial **8D (Eight Disciplines) Root Cause & Corrective Action**
    documentation directly from excursion detection alarms and root cause attribution engines.
    """)

    excursion_options = {
        "ETCH-07 Chamber C Edge-Ring Uniformity Excursion": {
            "target_id": "ETCH-07 / Chamber C",
            "yield_drop_pct": 7.98,
            "factor": "ETCH-07 / Chamber C (Electrostatic Chuck RF Gas Ring Non-Uniformity)",
            "evidence": "Contingency Fisher exact test p < 1e-12, Edge-Ring spatial pattern on 100% of chamber wafers."
        },
        "Photolithography Reticle RT-118B Repeating Defect": {
            "target_id": "LITHO Reticle RT-118B",
            "yield_drop_pct": 6.71,
            "factor": "RT-118B Reticle Field Defect at Die Position (1, 2)",
            "evidence": "80% systematic fallout at 3x3 exposure grid coordinate across Lots L262218-L262220."
        },
        "Frontend Implant NMOS Threshold Voltage (Vt_n) Shift": {
            "target_id": "Implant IMP-02 / Anneal ANN-01",
            "yield_drop_pct": 4.50,
            "factor": "WAT Vt_n Parameter Drift (+18 mV above nominal 312 mV)",
            "evidence": "WAT to sort VMIN/FMAX parametric failure correlation r = 0.62 across Lots L262226-L262231."
        }
    }

    selected_event = st.selectbox("Select Excursion Incident to Generate 8D Report:", list(excursion_options.keys()))
    event_data = excursion_options[selected_event]

    if st.button("📄 Generate Formal 8D Report", type="primary"):
        report_text = copilot.draft_8d_report(event_data, event_data)
        st.markdown(report_text)
        st.download_button(
            label="💾 Download 8D Report (Markdown)",
            data=report_text,
            file_name=f"8D_REPORT_{event_data['target_id'].replace(' ', '_').replace('/', '_')}.md",
            mime="text/markdown"
        )

with tab_docs:
    st.subheader("Indexed Specification Documents")
    st.markdown("""
    The Copilot RAG component operates strictly over locally verified semiconductor engineering documents.
    No data is ever transmitted to external endpoints or paid third-party LLM APIs.
    """)
    doc_summary = [
        {"Document": "AI_Yield_Advisor_Chip_Test_Technical_Design.pdf", "Pages": 19, "Scope": "Full Architecture, Math, Models, and Economic Sizing", "Status": "Indexed ✅"},
        {"Document": "technical_design_extracted.txt", "Pages": 19, "Scope": "Extracted Text Representation for High-Speed Retrieval", "Status": "Indexed ✅"},
        {"Document": "docs/TECHNICAL_DOCUMENTATION.md", "Pages": "Full Spec", "Scope": "System-level Integration and Data Pipeline", "Status": "Indexed ✅"},
        {"Document": "docs/MODEL_EVALUATION_REPORT.md", "Pages": "Benchmark", "Scope": "Official Model KPI Benchmarks and Performance", "Status": "Indexed ✅"},
    ]
    st.dataframe(pd.DataFrame(doc_summary), hide_index=True, use_container_width=True)
