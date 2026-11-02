"""
Data loader module for the AI-Optimized Yield Advisor.
Handles robust loading, caching, and querying of raw and processed datasets.
"""

import functools
import json
import logging
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np

import config

logger = logging.getLogger(__name__)


class DataLoader:
    """Central data access interface with memory optimization and caching."""

    def __init__(self, data_dir=config.DATA_DIR):
        self.data_dir = data_dir
        self._cache = {}

    def load_lots(self) -> pd.DataFrame:
        """Load lot summary metadata."""
        if "lots" not in self._cache:
            df = pd.read_csv(config.LOTS_CSV)
            df["fab_start_ts"] = pd.to_datetime(df["fab_start_ts"])
            df["sort_start_ts"] = pd.to_datetime(df["sort_start_ts"])
            self._cache["lots"] = df
        return self._cache["lots"].copy()

    def load_wafer_summary(self) -> pd.DataFrame:
        """Load wafer summary table with yields and timestamps."""
        if "wafer_summary" not in self._cache:
            df = pd.read_csv(config.WAFER_SUMMARY_CSV)
            df["sort_start_ts"] = pd.to_datetime(df["sort_start_ts"])
            self._cache["wafer_summary"] = df
        return self._cache["wafer_summary"].copy()

    def load_test_limits(self) -> pd.DataFrame:
        """Load test program limits and execution specs."""
        if "test_limits" not in self._cache:
            df = pd.read_csv(config.TEST_LIMITS_CSV)
            self._cache["test_limits"] = df
        return self._cache["test_limits"].copy()

    def load_bin_definitions(self) -> pd.DataFrame:
        """Load soft-to-hard bin mapping table."""
        if "bin_defs" not in self._cache:
            df = pd.read_csv(config.BIN_DEFINITIONS_CSV)
            self._cache["bin_defs"] = df
        return self._cache["bin_defs"].copy()

    def load_lot_context_mes(self) -> pd.DataFrame:
        """Load MES manufacturing genealogy for lots and wafers."""
        if "lot_context" not in self._cache:
            df = pd.read_csv(config.LOT_CONTEXT_MES_CSV)
            df["ts_in"] = pd.to_datetime(df["ts_in"])
            df["ts_out"] = pd.to_datetime(df["ts_out"])
            self._cache["lot_context"] = df
        return self._cache["lot_context"].copy()

    def load_wat_pcm(self) -> pd.DataFrame:
        """Load fab e-test / parametric wafer acceptance testing data."""
        if "wat_pcm" not in self._cache:
            df = pd.read_csv(config.WAT_PCM_CSV)
            self._cache["wat_pcm"] = df
        return self._cache["wat_pcm"].copy()

    def load_final_test_burnin(self) -> pd.DataFrame:
        """Load packaged die final test and burn-in reliability records."""
        if "final_test" not in self._cache:
            df = pd.read_csv(config.FINAL_TEST_BURNIN_CSV)
            self._cache["final_test"] = df
        return self._cache["final_test"].copy()

    def load_sort_die_results(self, lot_id: Optional[str] = None, wafer_id: Optional[int] = None) -> pd.DataFrame:
        """
        Load wafer sort per-die testing records.
        Optional filtering by lot_id and wafer_id to avoid reading entire 500k rows if not needed.
        """
        dtypes = {
            "lot_id": "category",
            "wafer_id": "int16",
            "die_x": "int16",
            "die_y": "int16",
            "site": "int8",
            "stage": "category",
            "insertion": "int8",
            "tester_id": "category",
            "probe_card_id": "category",
            "program_rev": "category",
            "temp_c": "int8",
            "hard_bin": "int16",
            "soft_bin": "int16",
            "pass": "int8",
            "test_time_ms": "float32",
            "CONTACT_RES": "float32",
            "IDDQ": "float32",
            "LEAKAGE": "float32",
            "SCAN_FAIL_COUNT": "float32",
            "MBIST_FAIL": "float32",
            "PLL_LOCK": "float32",
            "VMIN": "float32",
            "FMAX": "float32",
            "RING_OSC": "float32",
            "ADC_INL": "float32",
        }
        df = pd.read_csv(config.SORT_DIE_RESULTS_CSV, dtype=dtypes)
        if lot_id:
            df = df[df["lot_id"] == lot_id]
        if wafer_id:
            df = df[df["wafer_id"] == wafer_id]
        return df

    def get_final_disposition_die(self, sort_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Apply semiconductor disposition rule:
        When a die has insertion = 2 (retest), insertion 2 supersedes insertion 1.
        Returns deduplicated die records.
        """
        if sort_df is None:
            sort_df = self.load_sort_die_results()
        
        # Sort by insertion ascending so insertion 2 is last, then keep last
        final_df = sort_df.sort_values("insertion").drop_duplicates(
            subset=["lot_id", "wafer_id", "die_x", "die_y"], keep="last"
        ).reset_index(drop=True)
        return final_df

    def load_ground_truth_wafer_labels(self) -> pd.DataFrame:
        """
        Load wafer ground-truth labels.
        CRITICAL: Use ONLY for model evaluation and validation, NEVER as model feature inputs.
        """
        return pd.read_csv(config.GT_WAFER_LABELS_CSV)

    def load_ground_truth_die_labels(self) -> pd.DataFrame:
        """
        Load die ground-truth labels.
        CRITICAL: Use ONLY for model evaluation and validation, NEVER as model feature inputs.
        """
        return pd.read_csv(config.GT_DIE_LABELS_CSV)

    def load_ground_truth_injected_events(self) -> pd.DataFrame:
        """Load ground-truth injected events catalog."""
        return pd.read_csv(config.GT_INJECTED_EVENTS_CSV)

    def load_dataset_stats(self) -> Dict:
        """Load dataset summary statistics dictionary."""
        with open(config.DATASET_STATS_JSON, "r") as f:
            return json.load(f)
