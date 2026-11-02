"""
UI Helpers and Cached Data/Model Providers for Streamlit Application.
Ensures sub-second page transitions, shared state management, and consistent styling.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib

import config
from src.data_loader import DataLoader
from src.wafer_map import WaferMapClassifier
from src.excursion_detection import ExcursionDetector
from src.root_cause import RootCauseEngine
from src.latent_defect import LatentDefectEngine
from src.test_optimizer import TestTimeOptimizer
from src.yield_prediction import YieldPredictor
from src.recommendation_engine import RecommendationEngine


@st.cache_resource
def get_data_loader() -> DataLoader:
    return DataLoader()


@st.cache_data
def get_wafer_summary() -> pd.DataFrame:
    loader = get_data_loader()
    return loader.load_wafer_summary()


@st.cache_data
def get_lots() -> pd.DataFrame:
    loader = get_data_loader()
    return loader.load_lots()


@st.cache_data
def get_test_limits() -> pd.DataFrame:
    loader = get_data_loader()
    return loader.load_test_limits()


@st.cache_data
def get_bin_definitions() -> pd.DataFrame:
    loader = get_data_loader()
    return loader.load_bin_definitions()


@st.cache_data
def get_wafer_features() -> pd.DataFrame:
    if config.PREPROCESSED_SUMMARY_PATH.exists():
        return pd.read_parquet(config.PREPROCESSED_SUMMARY_PATH)
    from src.preprocessing import Preprocessor
    preprocessor = Preprocessor(get_data_loader())
    return preprocessor.build_wafer_features_dataset()


@st.cache_resource
def get_wafer_map_model():
    if config.WAFER_PATTERN_MODEL_PATH.exists():
        return WaferMapClassifier.load()
    return None


@st.cache_resource
def get_excursion_detector():
    if config.EXCURSION_DETECTOR_PATH.exists():
        return ExcursionDetector.load()
    return None


@st.cache_resource
def get_root_cause_engine():
    if config.ROOT_CAUSE_ENGINE_PATH.exists():
        return RootCauseEngine.load()
    return None


@st.cache_resource
def get_latent_defect_engine():
    if config.LATENT_DEFECT_ENGINE_PATH.exists():
        return LatentDefectEngine.load()
    return None


@st.cache_resource
def get_test_optimizer():
    if config.TEST_OPTIMIZER_PATH.exists():
        return TestTimeOptimizer.load()
    return None


@st.cache_resource
def get_yield_predictor():
    if config.YIELD_PREDICTION_MODEL_PATH.exists():
        return YieldPredictor.load()
    return None


@st.cache_resource
def get_copilot_engine():
    from src.copilot_engine import CopilotEngine
    return CopilotEngine.load()


@st.cache_resource
def get_recommendation_engine():
    rec_engine = RecommendationEngine()
    # Populate with pre-saved evaluation data
    eval_file = config.REPORTS_DIR / "model_evaluation_report.json"
    if eval_file.exists():
        import json
        with open(eval_file, "r") as f:
            rep = json.load(f)
        from src.preprocessing import Preprocessor
        preprocessor = Preprocessor(get_data_loader())
        retest_res = preprocessor.analyze_retest_recovery()
        detector = get_excursion_detector()
        lot_charts = detector.compute_lot_yield_charts(get_wafer_summary(), get_lots()) if detector else pd.DataFrame()
        events = detector.identify_excursion_events(lot_charts, get_wafer_features()) if detector else []
        rec_engine.generate_all_recommendations(
            excursion_events=events,
            root_causes=rep.get("engine_3_root_cause", []),
            latent_results=rep.get("engine_4_latent_defect", {}),
            test_proposals=rep.get("engine_5_test_optimizer", {}),
            retest_results=retest_res
        )
    return rec_engine


def apply_custom_css():
    """Apply professional semiconductor analytics dashboard styling."""
    st.markdown(
        """
        <style>
            /* Main container typography and padding */
            .main .block-container {
                padding-top: 1.5rem;
                padding-bottom: 2rem;
            }
            /* KPI Card styling */
            .kpi-card {
                background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
                border: 1px solid #334155;
                border-radius: 8px;
                padding: 16px;
                margin-bottom: 12px;
                box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2);
            }
            .kpi-title {
                font-size: 0.85rem;
                color: #94a3b8;
                text-transform: uppercase;
                letter-spacing: 0.05em;
                margin-bottom: 4px;
            }
            .kpi-value {
                font-size: 1.8rem;
                font-weight: 700;
                color: #f8fafc;
            }
            .kpi-delta {
                font-size: 0.8rem;
                font-weight: 600;
            }
            .kpi-delta.good { color: #10b981; }
            .kpi-delta.warn { color: #f59e0b; }
            .kpi-delta.bad { color: #ef4444; }
            
            /* Status badges */
            .badge-critical {
                background-color: #ef4444; color: white; padding: 3px 8px; border-radius: 4px; font-weight: 600; font-size: 0.75rem;
            }
            .badge-high {
                background-color: #f97316; color: white; padding: 3px 8px; border-radius: 4px; font-weight: 600; font-size: 0.75rem;
            }
            .badge-medium {
                background-color: #eab308; color: black; padding: 3px 8px; border-radius: 4px; font-weight: 600; font-size: 0.75rem;
            }
            .badge-low {
                background-color: #3b82f6; color: white; padding: 3px 8px; border-radius: 4px; font-weight: 600; font-size: 0.75rem;
            }
        </style>
        """,
        unsafe_allow_html=True
    )


def render_header(title: str, subtitle: str):
    """Render consistent page header."""
    apply_custom_css()
    st.markdown(
        f"""
        <div style="border-bottom: 2px solid #3b82f6; padding-bottom: 12px; margin-bottom: 20px;">
            <div style="display: flex; justify-content: space-between; align-items: baseline;">
                <div>
                    <h1 style="margin: 0; font-size: 2.1rem; color: #1e3a8a; font-weight: 800;">{title}</h1>
                    <p style="margin: 4px 0 0 0; color: #475569; font-size: 1.05rem; font-weight: 500;">{subtitle}</p>
                </div>
                <div style="text-align: right;">
                    <span style="background: #dbeafe; color: #1e40af; padding: 4px 10px; border-radius: 12px; font-size: 0.8rem; font-weight: 700;">
                        PRODUCT: AX7-SOC | ATE: AX7_CP1_R4.2
                    </span>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
