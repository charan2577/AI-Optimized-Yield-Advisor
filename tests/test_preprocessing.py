"""
Unit tests for preprocessing, coordinate normalization, retest disposition, and feature extraction.
"""

import pytest
import numpy as np
import pandas as pd

import config
from src.data_loader import DataLoader
from src.preprocessing import Preprocessor


@pytest.fixture
def loader():
    return DataLoader()


@pytest.fixture
def preprocessor(loader):
    return Preprocessor(loader)


def test_coordinate_normalization():
    """Verify radial distance and polar angle computation."""
    die_x = np.array([12, 0, 25])
    die_y = np.array([12, 12, 12])
    r, theta = Preprocessor.compute_spatial_coordinates(die_x, die_y)
    
    # Center die (12, 12) is close to radius 0
    assert r[0] < 0.1
    # Edge dies have radius close to 1.0
    assert r[1] > 0.95
    assert r[2] > 0.95


def test_retest_disposition(loader):
    """Verify final disposition logic overrides insertion 1 with insertion 2."""
    sample = loader.load_sort_die_results(lot_id="L262200")
    final_dispo = loader.get_final_disposition_die(sample)
    
    # Check no duplicate die keys exist
    dups = final_dispo.duplicated(subset=["lot_id", "wafer_id", "die_x", "die_y"]).sum()
    assert dups == 0
    # Every wafer must have exactly 500 dies
    wafer_counts = final_dispo.groupby("wafer_id")["die_x"].count()
    assert (wafer_counts == 500).all()


def test_spatial_feature_extraction(loader, preprocessor):
    """Verify spatial feature extraction produces expected metrics."""
    sample = loader.load_sort_die_results(lot_id="L262200", wafer_id=1)
    feats = preprocessor.extract_wafer_spatial_features(sample)
    
    assert "fail_rate" in feats
    assert "center_density" in feats
    assert "edge_density" in feats
    assert "reticle_pattern_score" in feats
    assert "scratch_line_score" in feats
    assert "circle_cluster_score" in feats
    assert "circle_cluster_ratio" in feats
    assert "scratch_line_ratio" in feats
    assert "scratch_aspect_ratio" in feats
    assert "max_continuous_cluster" in feats
    assert 0.0 <= feats["fail_rate"] <= 1.0
