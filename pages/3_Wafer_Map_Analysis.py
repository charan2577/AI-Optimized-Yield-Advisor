"""
PAGE 3: WAFER MAP ANALYSIS.
Interactive 26x26 spatial wafer map viewer, AI pattern classifier, and ground-truth validation.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

import config
from src.ui_helpers import (
    render_header,
    get_data_loader,
    get_wafer_map_model,
    get_wafer_features,
)
from src.preprocessing import Preprocessor

st.set_page_config(page_title="Wafer Map Analysis | Yield Advisor", layout="wide")
render_header(
    title="Wafer Map Spatial Pattern Intelligence",
    subtitle="Interactive Die-Level Spatial Inspection, Signature Classification, and Anomaly Localization"
)

loader = get_data_loader()
ws = loader.load_wafer_summary()
model = get_wafer_map_model()
wafer_features_df = get_wafer_features()

# Selection Controls
col_sel1, col_sel2, col_sel3 = st.columns([2, 2, 2])

with col_sel1:
    lots_list = sorted(list(ws["lot_id"].unique()))
    selected_lot = st.selectbox("Select Manufacturing Lot", lots_list, index=0)

with col_sel2:
    available_wafers = sorted(list(ws[ws["lot_id"] == selected_lot]["wafer_id"].unique()))
    selected_wafer = st.selectbox("Select Wafer ID", available_wafers, index=0)

with col_sel3:
    eval_mode = st.toggle("🧪 Enable Ground-Truth Evaluation Mode", value=False)

# Load Dies for Selected Wafer
with st.spinner("Loading die testing results..."):
    dies = loader.load_sort_die_results(lot_id=selected_lot, wafer_id=selected_wafer)

# Apply final disposition
final_dies = loader.get_final_disposition_die(dies)
gross_count = len(final_dies)
good_count = int(final_dies["pass"].sum())
fail_count = gross_count - good_count
yield_pct = (good_count / gross_count) * 100 if gross_count > 0 else 0.0

# Extract Features and Classify
preprocessor = Preprocessor(loader)
features = preprocessor.extract_wafer_spatial_features(dies)
pred_class, confidence, probs = model.predict_wafer(features) if model else ("Unknown", 0.0, {})

# Display KPI Row
k1, k2, k3, k4, k5 = st.columns(5)
with k1:
    st.metric("Wafer Yield", f"{yield_pct:.2f}%", delta=f"{good_count}/{gross_count} Good")
with k2:
    st.metric("Failing Dies", f"{fail_count}", delta=f"{100-yield_pct:.2f}% Loss", delta_color="inverse")
with k3:
    st.metric("AI Predicted Pattern", pred_class)
with k4:
    st.metric("Model Confidence", f"{confidence*100:.1f}%")
with k5:
    if eval_mode:
        gt_wafers = loader.load_ground_truth_wafer_labels()
        gt_match = gt_wafers[(gt_wafers["lot_id"] == selected_lot) & (gt_wafers["wafer_id"] == selected_wafer)]
        gt_label = gt_match["pattern_labels"].iloc[0] if len(gt_match) > 0 else "N/A"
        is_match = pred_class in gt_label or (pred_class == "Clean" and gt_label == "Clean")
        st.metric("Ground Truth Label", gt_label, delta="MATCH ✅" if is_match else "DISCREPANCY ⚠️")
    else:
        st.metric("Evaluation Status", "Operational (Inference)")

st.markdown("---")

col_map, col_details = st.columns([3, 2])

with col_map:
    st.subheader(f"🗺️ Wafer Map: {selected_lot} - Wafer #{selected_wafer:02d}")
    
    view_mode = st.radio("Display Mode:", ["Pass / Fail Status", "Soft-Bin Categories"], horizontal=True)

    # Prepare 26x26 grid representation
    grid_data = np.full((config.DIE_GRID_SIZE, config.DIE_GRID_SIZE), np.nan)
    text_data = np.full((config.DIE_GRID_SIZE, config.DIE_GRID_SIZE), "", dtype=object)

    bin_names = config.SOFT_BIN_NAMES

    for _, row in final_dies.iterrows():
        x = int(row["die_x"])
        y = int(row["die_y"])
        sb = int(row["soft_bin"])
        p = int(row["pass"])
        sb_name = bin_names.get(sb, f"SB{sb}")
        
        if view_mode == "Pass / Fail Status":
            grid_data[y, x] = 1 if p == 1 else 0
        else:
            grid_data[y, x] = sb

        v_val = row.get("VMIN", "N/A")
        f_val = row.get("FMAX", "N/A")
        i_val = row.get("IDDQ", "N/A")
        tt_val = row.get("test_time_ms", "N/A")
        text_data[y, x] = f"Die ({x},{y})<br>Site: {row['site']}<br>Bin: {sb_name}<br>Pass: {bool(p)}<br>IDDQ: {i_val} uA<br>Vmin: {v_val} V<br>Fmax: {f_val} MHz<br>Test Time: {tt_val} ms"

    if view_mode == "Pass / Fail Status":
        fig_map = go.Figure(data=go.Heatmap(
            z=grid_data,
            text=text_data,
            hoverinfo="text",
            colorscale=[[0, "#ef4444"], [1, "#10b981"]],
            showscale=False,
            xgap=1,
            ygap=1,
        ))
    else:
        fig_map = go.Figure(data=go.Heatmap(
            z=grid_data,
            text=text_data,
            hoverinfo="text",
            colorscale="Viridis",
            showscale=True,
            colorbar=dict(title="Soft Bin ID"),
            xgap=1,
            ygap=1,
        ))

    fig_map.update_layout(
        width=520,
        height=520,
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis=dict(title="Die X (0 to 25)", showgrid=False, zeroline=False),
        yaxis=dict(title="Die Y (0 to 25)", autorange="reversed", showgrid=False, zeroline=False),
    )
    st.plotly_chart(fig_map, width="content")

with col_details:
    st.subheader("🤖 AI Spatial Classifier Probabilities")
    
    if probs:
        p_df = pd.DataFrame(list(probs.items()), columns=["Pattern", "Probability"]).sort_values("Probability", ascending=True)
        fig_prob = px.bar(
            p_df,
            x="Probability",
            y="Pattern",
            orientation="h",
            color="Probability",
            color_continuous_scale="Blues",
            text="Probability",
            range_x=[0, 1.05]
        )
        fig_prob.update_traces(texttemplate="%{text:.1%}", textposition="outside")
        fig_prob.update_layout(height=260, margin=dict(l=10, r=20, t=10, b=10), showlegend=False)
        st.plotly_chart(fig_prob, width="stretch")

    st.subheader("📐 Extracted Spatial Metrics")
    feat_view = {
        "Radial Gradient (Edge - Center)": round(features.get("radial_gradient", 0), 3),
        "Center Density": round(features.get("center_density", 0), 3),
        "Edge Density": round(features.get("edge_density", 0), 3),
        "Reticle Repeating Score": round(features.get("reticle_pattern_score", 0), 3),
        "Scratch Line Score": round(features.get("scratch_line_score", 0), 1),
        "Circle Cluster Score": round(features.get("circle_cluster_score", 0), 1),
        "Centroid Offset (X, Y)": f"({features.get('x_mean', 0):.2f}, {features.get('y_mean', 0):.2f})",
    }
    st.table(pd.DataFrame(list(feat_view.items()), columns=["Metric", "Value"]))
