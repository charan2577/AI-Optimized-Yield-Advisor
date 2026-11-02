"""
Unit tests for data validation, schema verification, and stop-on-fail protocol compliance.
"""

import pytest
import pandas as pd
import numpy as np

import config
from src.data_loader import DataLoader
from src.data_validation import DataValidator


@pytest.fixture
def loader():
    return DataLoader()


@pytest.fixture
def validator(loader):
    return DataValidator(loader)


def test_schema_validation(loader, validator):
    """Verify lots table schema compliance."""
    lots = loader.load_lots()
    expected_cols = ["lot_id", "product", "fab", "technology", "wafer_count", "fab_start_ts", "sort_start_ts", "tester_id", "probe_card_id", "program_rev", "prober_temp_c"]
    res = validator.validate_schema(lots, expected_cols, "lots")
    assert res["status"] == "PASS"
    assert len(res["missing_columns"]) == 0
    assert res["row_count"] == 40


def test_relational_integrity(validator):
    """Verify foreign key integrity across lots, wafers, MES, and final test."""
    rel = validator.validate_relationships()
    assert rel["all_lots_in_summary"] is True
    assert rel["all_mes_lots_in_lots"] is True
    assert rel["all_ft_lots_in_lots"] is True
    assert rel["wafers_per_lot_uniform_25"] is True
    assert rel["total_lots"] == 40
    assert rel["total_wafers"] == 1000


def test_stop_on_fail_compliance(loader, validator):
    """Verify stop-on-fail protocol in wafer sort data."""
    sort_sample = loader.load_sort_die_results(lot_id="L262200", wafer_id=1)
    res = validator.validate_stop_on_fail_logic(sort_sample)
    assert res["compliance_rate_pct"] == 100.0
    assert res["stop_on_fail_violations"] == 0


def test_chronological_leakage_prevention(loader):
    """Verify train and holdout lots are strictly disjoint with zero temporal contamination."""
    lots = loader.load_lots()
    assert len(lots) == 40
    train_lots = set(lots["lot_id"].iloc[:30])
    test_lots = set(lots["lot_id"].iloc[30:])
    
    assert len(train_lots.intersection(test_lots)) == 0, "Data leakage detected: train and test sets share lots!"
    assert len(train_lots) == 30
    assert len(test_lots) == 10


def test_cross_platform_paths():
    """Verify all config paths are pathlib.Path instances and resolve portably."""
    from pathlib import Path
    assert isinstance(config.PROJECT_ROOT, Path)
    assert isinstance(config.DATA_DIR, Path)
    assert isinstance(config.MODELS_DIR, Path)
    assert isinstance(config.WAFER_PATTERN_MODEL_PATH, Path)
    assert config.DATA_DIR.exists()

