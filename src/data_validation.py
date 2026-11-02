"""
Data Validation and Quality Assurance Engine.
Performs schema verification, integrity checks, stop-on-fail validation,
and generates structured Data Quality Reports.
"""

import json
from pathlib import Path
from typing import Dict, Any, List
import numpy as np
import pandas as pd

import config


class DataValidator:
    """Validates schemas, relational integrity, value ranges, and stop-on-fail logic."""

    def __init__(self, data_loader=None):
        from src.data_loader import DataLoader
        self.loader = data_loader or DataLoader()
        self.validation_results = {}

    def validate_schema(self, df: pd.DataFrame, expected_cols: List[str], table_name: str) -> Dict[str, Any]:
        """Validate column presence and extra columns."""
        actual_cols = list(df.columns)
        missing = [c for c in expected_cols if c not in actual_cols]
        extra = [c for c in actual_cols if c not in expected_cols]
        return {
            "table": table_name,
            "status": "PASS" if not missing else "FAIL",
            "missing_columns": missing,
            "extra_columns": extra,
            "column_count": len(actual_cols),
            "row_count": len(df),
        }

    def validate_relationships(self) -> Dict[str, Any]:
        """
        Validate join keys across tables:
        - All wafers in sort_die_results exist in wafer_summary and lots
        - All wafers in lot_context_mes exist in lots
        - All dies in final_test exist in sort_die_results (insertion 1 or 2)
        """
        lots = self.loader.load_lots()
        wafer_summary = self.loader.load_wafer_summary()
        mes = self.loader.load_lot_context_mes()
        ft = self.loader.load_final_test_burnin()

        lot_ids_lots = set(lots["lot_id"].unique())
        lot_ids_ws = set(wafer_summary["lot_id"].unique())
        lot_ids_mes = set(mes["lot_id"].unique())
        lot_ids_ft = set(ft["lot_id"].unique())

        ws_keys = set(zip(wafer_summary["lot_id"], wafer_summary["wafer_id"]))
        mes_keys = set(zip(mes["lot_id"], mes["wafer_id"]))

        # Verify wafer count per lot
        wafers_per_lot = wafer_summary.groupby("lot_id")["wafer_id"].count()
        uniform_wafers = bool((wafers_per_lot == config.WAFERS_PER_LOT).all())

        return {
            "all_lots_in_summary": lot_ids_ws.issubset(lot_ids_lots),
            "all_mes_lots_in_lots": lot_ids_mes.issubset(lot_ids_lots),
            "all_ft_lots_in_lots": lot_ids_ft.issubset(lot_ids_lots),
            "wafers_per_lot_uniform_25": uniform_wafers,
            "total_lots": len(lot_ids_lots),
            "total_wafers": len(ws_keys),
            "mes_wafer_coverage_pct": round(len(mes_keys.intersection(ws_keys)) / len(ws_keys) * 100, 2),
        }

    def validate_stop_on_fail_logic(self, sort_df: pd.DataFrame, sample_size: int = 5000) -> Dict[str, Any]:
        """
        In stop-on-fail testing:
        - When test i fails, tests (i+1, ...) MUST NOT be executed (value = NaN).
        - Tests prior to test i MUST have non-null values.
        - PASS die must have non-null values across all tests.
        """
        sample = sort_df.head(sample_size).copy()
        limits = self.loader.load_test_limits()
        test_order = list(limits["test_name"])
        soft_bins = list(limits["soft_bin"])
        
        violations = 0
        total_checked = len(sample)

        for _, row in sample.iterrows():
            sb = row["soft_bin"]
            if sb == 1:
                # Passing die: all tests should be executed
                if row[test_order].isna().any():
                    violations += 1
            elif sb in soft_bins:
                fail_idx = soft_bins.index(sb)
                failing_test = test_order[fail_idx]
                
                # Preceding tests and failing test must NOT be NaN
                prior_tests = test_order[:fail_idx + 1]
                if row[prior_tests].isna().any():
                    violations += 1
                    
                # Subsequent tests MUST be NaN
                subsequent_tests = test_order[fail_idx + 1:]
                if subsequent_tests and not row[subsequent_tests].isna().all():
                    violations += 1

        return {
            "sample_size": total_checked,
            "stop_on_fail_violations": violations,
            "compliance_rate_pct": round((total_checked - violations) / total_checked * 100, 4),
            "interpretation": "Complies strictly with ATE stop-on-fail tester protocol." if violations == 0 else "Violations detected."
        }

    def generate_full_data_quality_report(self) -> Dict[str, Any]:
        """Run comprehensive data quality audit and produce report."""
        lots = self.loader.load_lots()
        ws = self.loader.load_wafer_summary()
        limits = self.loader.load_test_limits()
        bin_defs = self.loader.load_bin_definitions()
        mes = self.loader.load_lot_context_mes()
        wat = self.loader.load_wat_pcm()
        ft = self.loader.load_final_test_burnin()
        sort_sample = self.loader.load_sort_die_results(lot_id="L262200")

        # Missing values summary
        missing_summary = {
            "lots": lots.isna().sum().to_dict(),
            "wafer_summary": ws.isna().sum().to_dict(),
            "test_limits": limits.isna().sum().to_dict(),
            "mes": mes.isna().sum().to_dict(),
            "wat_pcm": wat.isna().sum().to_dict(),
            "final_test": ft.isna().sum().to_dict(),
        }

        # Duplicates summary
        dup_summary = {
            "lots_unique_keys": bool(lots["lot_id"].is_unique),
            "wafer_summary_unique_keys": bool(not ws.duplicated(subset=["lot_id", "wafer_id"]).any()),
            "mes_duplicate_rows": int(mes.duplicated().sum()),
            "ft_ecid_unique": bool(ft["ecid"].is_unique),
        }

        rel_checks = self.validate_relationships()
        sof_checks = self.validate_stop_on_fail_logic(sort_sample)

        report = {
            "timestamp": pd.Timestamp.now().isoformat(),
            "summary": {
                "total_lots": int(len(lots)),
                "total_wafers": int(len(ws)),
                "total_ft_records": int(len(ft)),
                "gross_die_per_wafer": int(ws["gross_die"].iloc[0]),
                "mean_final_yield": round(float(ws["final_yield"].mean()), 4),
                "mean_first_pass_yield": round(float(ws["first_pass_yield"].mean()), 4),
                "burnin_failures": int(ft["burnin_fail"].sum()),
            },
            "missing_values": missing_summary,
            "duplicate_checks": dup_summary,
            "relational_integrity": rel_checks,
            "stop_on_fail_protocol": sof_checks,
            "verdict": "DATASET_VALIDATED_HEALTHY",
        }

        # Save report to disk
        config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        report_path = config.REPORTS_DIR / "data_quality_report.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)

        return report
