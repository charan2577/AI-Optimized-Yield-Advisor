"""
Unit tests for AI Engine inference, model loading, and predictions.
"""

import pytest
import numpy as np
import pandas as pd

import config
from src.wafer_map import WaferMapClassifier
from src.excursion_detection import ExcursionDetector
from src.root_cause import RootCauseEngine
from src.latent_defect import LatentDefectEngine
from src.test_optimizer import TestTimeOptimizer
from src.yield_prediction import YieldPredictor
from src.data_loader import DataLoader


def test_wafer_map_model_inference():
    """Verify wafer map classifier loads and predicts valid classes without feature-name warnings."""
    import warnings
    assert config.WAFER_PATTERN_MODEL_PATH.exists()
    model = WaferMapClassifier.load()
    assert model.is_trained is True

    # Test dummy clean wafer feature vector
    clean_feats = {
        "fail_rate": 0.05,
        "center_density": 0.04,
        "donut_density": 0.05,
        "edge_density": 0.06,
        "radial_gradient": 0.01,
        "center_to_edge_ratio": 0.8,
        "edge_to_center_ratio": 1.2,
        "x_mean": 0.0,
        "y_mean": 0.0,
        "spatial_std": 6.0,
        "cluster_density_max": 0.1,
        "reticle_pattern_score": 0.02,
        "angular_variance": 0.1,
        "scratch_line_score": 4.0,
        "circle_cluster_score": 5.0,
        "circle_cluster_ratio": 0.05,
        "scratch_line_ratio": 0.04,
        "scratch_aspect_ratio": 1.0,
        "max_continuous_cluster": 3.0,
    }

    # Verify zero UserWarnings during inference
    with warnings.catch_warnings(record=True) as recorded_warnings:
        warnings.simplefilter("always")
        pred, conf, probs = model.predict_wafer(clean_feats)

        user_warnings = [w for w in recorded_warnings if issubclass(w.category, UserWarning)]
        assert len(user_warnings) == 0, f"Unexpected UserWarning: {[str(w.message) for w in user_warnings]}"

    assert pred in model.classes_
    assert 0.0 <= conf <= 1.0


def test_wafer_map_feature_name_consistency():
    """Explicitly verify feature names and order match model.feature_names_in_."""
    model = WaferMapClassifier.load()
    assert hasattr(model.model, "feature_names_in_"), "RandomForest model has no feature_names_in_ attribute."
    expected_features = list(model.model.feature_names_in_)
    assert model.FEATURE_COLS == expected_features, "FEATURE_COLS does not match model.feature_names_in_ order."


def test_excursion_detector():
    """Verify excursion detector computes EWMA/CUSUM control charts."""
    assert config.EXCURSION_DETECTOR_PATH.exists()
    detector = ExcursionDetector.load()
    loader = DataLoader()
    ws = loader.load_wafer_summary()
    lots = loader.load_lots()
    
    lot_charts = detector.compute_lot_yield_charts(ws, lots)
    assert len(lot_charts) == 40
    assert "ewma_alarm" in lot_charts.columns
    assert "cusum_alarm" in lot_charts.columns
    assert lot_charts["is_excursion_lot"].sum() > 0


def test_latent_defect_dpat_limits():
    """Verify robust AEC-Q001 PAT limit calculations."""
    vals = np.array([50.0, 51.0, 49.5, 50.5, 52.0, 48.0, 50.2, 51.5, 50.8, 49.2, 51.1, 50.4, 63.0])
    low, high, med, sig = LatentDefectEngine.compute_dpat_limits(vals, k=6.0, lsl=40.0, usl=65.0)
    assert low >= 40.0
    assert high <= 65.0
    assert 49.0 <= med <= 52.0


def test_test_time_optimizer():
    """Verify test optimizer generates valid proposals and escape bounds."""
    assert config.TEST_OPTIMIZER_PATH.exists()
    optimizer = TestTimeOptimizer.load()
    redundancy_df = optimizer.analyze_test_redundancy()
    assert len(redundancy_df) == 10
    assert "escape_dppm_bound" in redundancy_df.columns
    assert (redundancy_df["escape_dppm_bound"] > 0).all()

    # Verify data-driven proposal generation
    proposals = optimizer.evaluate_optimization_proposals(redundancy_df)
    assert "proposals" in proposals
    assert len(proposals["proposals"]) >= 2
    for p in proposals["proposals"]:
        assert p["status"] == "REQUIRES_CCB_APPROVAL"
        assert p["escape_risk_dppm"] > 0
        assert p["test_time_saved_ms"] > 0
        assert "Advisory" in p.get("governance", "")


def test_root_cause_prioritization_and_distinction():
    """Verify RootCauseEngine isolates physical causes and distinguishes statistical associations."""
    assert config.ROOT_CAUSE_ENGINE_PATH.exists()
    rce = RootCauseEngine.load()
    from src.preprocessing import Preprocessor
    loader = DataLoader()
    p = Preprocessor(loader)
    df = p.build_wafer_features_dataset()
    causes = rce.generate_ranked_root_causes(df)
    assert len(causes) >= 3

    top3_factors = [c["factor"] for c in causes[:3]]
    # All 3 top causes must match confirmed physical events
    assert any("RT-118B" in f for f in top3_factors), "RT-118B reticle not in top 3"
    assert any("ETCH-07" in f for f in top3_factors), "ETCH-07 chamber excursion not in top 3"
    assert any("Vt_n" in f for f in top3_factors), "Vt_n parameter drift not in top 3"

    # Confirm attribution_type labeling
    confirmed_causes = [c for c in causes if "Confirmed Physical" in c["attribution_type"]]
    statistical_causes = [c for c in causes if "Statistical Equipment" in c["attribution_type"]]
    assert len(confirmed_causes) >= 2
    assert len(statistical_causes) >= 1



def test_yield_predictor_inference():
    """Verify yield predictor loads and generates quantile predictions."""
    assert config.YIELD_PREDICTION_MODEL_PATH.exists()
    predictor = YieldPredictor.load()
    assert predictor.is_trained is True

    loader = DataLoader()
    ws = loader.load_wafer_summary()
    assert len(ws) == 1000

    row = pd.DataFrame([{
        "vt_n_mean": 312.5,
        "vt_p_mean": -298.0,
        "idsat_n_mean": 1080.0,
        "rs_poly_mean": 7.8,
        "contact_chain_mean": 41.0,
        "ETCH_equipment_id": "ETCH-05",
        "ETCH_chamber": "A",
        "CMP_equipment_id": "CMP-01",
        "LITHO_reticle_id": "RT-118A",
        "WAFER_SORT_LOAD_equipment_id": "LOAD_v1",
        "probe_card_id": "PC-4410",
        "wafer_id": 1,
    }])
    p10, p50, p90 = predictor.predict_wafer_yield(row)
    assert 0.40 <= p10 <= p50 <= p90 <= 1.0


def test_copilot_engine_rag():
    """Verify offline RAG copilot retrieves citations and answers queries."""
    from src.copilot_engine import CopilotEngine
    engine = CopilotEngine()
    engine.build_index()
    assert engine.is_indexed is True
    assert len(engine.chunks) > 0

    res = engine.query("What caused the ETCH-07 Chamber C excursion?")
    assert res["grounded"] is True
    assert len(res["sources"]) > 0
    assert "ETCH-07" in res["answer"]
