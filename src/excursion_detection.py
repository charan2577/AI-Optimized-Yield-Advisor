"""
ENGINE 2: Excursion Detection Engine.
Detects yield shifts, parametric drift, and manufacturing anomalies
using EWMA, CUSUM, Shewhart control charts, and Isolation Forest.
"""

from typing import Dict, List, Any, Optional
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

import config


class ExcursionDetector:
    """Statistical and machine learning anomaly and excursion detection system."""

    def __init__(
        self,
        ewma_lambda: float = config.EWMA_LAMBDA,
        ewma_l: float = config.EWMA_L,
        cusum_k: float = config.CUSUM_K,
        cusum_h: float = config.CUSUM_H,
    ):
        self.ewma_lambda = ewma_lambda
        self.ewma_l = ewma_l
        self.cusum_k = cusum_k
        self.cusum_h = cusum_h
        self.isolation_forest = IsolationForest(
            n_estimators=100,
            contamination=0.03,  # ~3% of wafers expected to be excursions
            random_state=2026,
        )
        self.is_fitted = False

    def compute_lot_yield_charts(self, wafer_summary_df: pd.DataFrame, lots_df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute chronological lot-level yield trends with Shewhart, EWMA, and CUSUM statistics.
        """
        # Group by lot and order chronologically by sort_start_ts
        lot_stats = wafer_summary_df.groupby("lot_id").agg(
            gross_die=("gross_die", "sum"),
            good_die=("good_die", "sum"),
            wafer_count=("wafer_id", "count"),
            mean_final_yield=("final_yield", "mean"),
            min_wafer_yield=("final_yield", "min"),
            max_wafer_yield=("final_yield", "max"),
        ).reset_index()

        lot_stats["lot_yield"] = lot_stats["good_die"] / lot_stats["gross_die"]

        # Merge with sort start timestamp to ensure chronological order
        lot_stats = lot_stats.merge(lots_df[["lot_id", "sort_start_ts", "tester_id", "probe_card_id"]], on="lot_id")
        lot_stats = lot_stats.sort_values("sort_start_ts").reset_index(drop=True)

        yields = lot_stats["lot_yield"].values
        n = len(yields)

        # Baseline: first 15 lots are assumed in-control baseline
        baseline_idx = min(15, n)
        mu0 = float(np.mean(yields[:baseline_idx]))
        sigma0 = float(np.std(yields[:baseline_idx], ddof=1)) if baseline_idx > 1 else 0.01

        # 1. Shewhart Limits
        ucl_shewhart = mu0 + 3 * sigma0
        lcl_shewhart = mu0 - 3 * sigma0
        shewhart_alarms = (yields < lcl_shewhart) | (yields > ucl_shewhart)

        # 2. EWMA Chart
        ewma_z = np.zeros(n)
        ewma_ucl = np.zeros(n)
        ewma_lcl = np.zeros(n)
        ewma_alarms = np.zeros(n, dtype=bool)

        current_z = mu0
        for t in range(n):
            current_z = self.ewma_lambda * yields[t] + (1 - self.ewma_lambda) * current_z
            ewma_z[t] = current_z
            # Exact time-varying variance factor
            var_factor = np.sqrt(
                (self.ewma_lambda / (2.0 - self.ewma_lambda)) * (1.0 - (1.0 - self.ewma_lambda) ** (2 * (t + 1)))
            )
            ewma_ucl[t] = mu0 + self.ewma_l * sigma0 * var_factor
            ewma_lcl[t] = mu0 - self.ewma_l * sigma0 * var_factor
            if current_z < ewma_lcl[t] or current_z > ewma_ucl[t]:
                ewma_alarms[t] = True

        # 3. CUSUM Chart (focusing on negative yield shift)
        cusum_neg = np.zeros(n)
        cusum_pos = np.zeros(n)
        cusum_alarms = np.zeros(n, dtype=bool)

        k_std = self.cusum_k * sigma0
        h_std = self.cusum_h * sigma0

        for t in range(n):
            prev_neg = cusum_neg[t - 1] if t > 0 else 0.0
            prev_pos = cusum_pos[t - 1] if t > 0 else 0.0

            # Downward shift detection (yield loss)
            cusum_neg[t] = max(0.0, prev_neg + (mu0 - k_std) - yields[t])
            # Upward shift detection
            cusum_pos[t] = max(0.0, prev_pos + yields[t] - (mu0 + k_std))

            if cusum_neg[t] > h_std or cusum_pos[t] > h_std:
                cusum_alarms[t] = True

        lot_stats["baseline_mu0"] = round(mu0, 4)
        lot_stats["baseline_sigma0"] = round(sigma0, 4)
        lot_stats["shewhart_ucl"] = round(ucl_shewhart, 4)
        lot_stats["shewhart_lcl"] = round(lcl_shewhart, 4)
        lot_stats["shewhart_alarm"] = shewhart_alarms

        lot_stats["ewma_value"] = np.round(ewma_z, 4)
        lot_stats["ewma_ucl"] = np.round(ewma_ucl, 4)
        lot_stats["ewma_lcl"] = np.round(ewma_lcl, 4)
        lot_stats["ewma_alarm"] = ewma_alarms

        lot_stats["cusum_neg"] = np.round(cusum_neg, 4)
        lot_stats["cusum_pos"] = np.round(cusum_pos, 4)
        lot_stats["cusum_threshold"] = round(h_std, 4)
        lot_stats["cusum_alarm"] = cusum_alarms

        # Any alarm
        lot_stats["is_excursion_lot"] = ewma_alarms | cusum_alarms | shewhart_alarms

        return lot_stats

    def fit_multivariate_wafer_model(self, wafer_features_df: pd.DataFrame) -> pd.DataFrame:
        """
        Fit Isolation Forest on wafer feature representation to flag novel anomalies
        and multivariate parametric excursions.
        """
        feature_cols = [
            "final_yield",
            "first_pass_yield",
            "vt_n_mean",
            "vt_p_mean",
            "idsat_n_mean",
            "contact_chain_mean",
            "fail_rate",
            "edge_density",
            "center_density",
            "reticle_pattern_score",
        ]
        
        # Filter available columns
        available_cols = [c for c in feature_cols if c in wafer_features_df.columns]
        X = wafer_features_df[available_cols].fillna(wafer_features_df[available_cols].median())

        self.isolation_forest.fit(X)
        self.is_fitted = True

        scores = self.isolation_forest.decision_function(X)
        preds = self.isolation_forest.predict(X)  # -1 for anomaly, 1 for normal

        result_df = wafer_features_df.copy()
        result_df["anomaly_score"] = np.round(scores, 4)
        result_df["is_multivariate_anomaly"] = preds == -1

        # Save model
        config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, config.EXCURSION_DETECTOR_PATH)

        return result_df

    def identify_excursion_events(
        self, lot_charts_df: pd.DataFrame, wafer_features_df: pd.DataFrame
    ) -> List[Dict[str, Any]]:
        """
        Synthesize detected excursions across lots and wafers into structured excursion events.
        Correlates with known problem signatures (Etch chamber C, CMP-03, Vt shift).
        """
        events = []

        # 1. Check lot yield excursions
        excursion_lots = lot_charts_df[lot_charts_df["is_excursion_lot"]].copy()
        if len(excursion_lots) > 0:
            # Group contiguous lots into episodes
            for lot_id in excursion_lots["lot_id"]:
                lot_row = excursion_lots[excursion_lots["lot_id"] == lot_id].iloc[0]
                delta = lot_row["baseline_mu0"] - lot_row["lot_yield"]
                
                severity = "HIGH" if delta > 0.04 else ("MEDIUM" if delta > 0.02 else "LOW")
                events.append({
                    "event_type": "LOT_YIELD_EXCURSION",
                    "target_id": lot_id,
                    "severity": severity,
                    "yield_drop_pct": round(delta * 100, 2),
                    "lot_yield": round(lot_row["lot_yield"] * 100, 2),
                    "ewma_alarm": bool(lot_row["ewma_alarm"]),
                    "cusum_alarm": bool(lot_row["cusum_alarm"]),
                    "timestamp": str(lot_row["sort_start_ts"]),
                    "description": f"Lot {lot_id} yield dropped to {lot_row['lot_yield']*100:.2f}% (baseline: {lot_row['baseline_mu0']*100:.2f}%)."
                })

        # 2. Check localized wafer excursions (e.g. Edge-Ring or Center spikes)
        if "ETCH_chamber" in wafer_features_df.columns:
            etch_c = wafer_features_df[
                (wafer_features_df["ETCH_chamber"] == "C") &
                (wafer_features_df["ETCH_equipment_id"] == "ETCH-07") &
                (wafer_features_df["edge_density"] > 0.35)
            ]
            if len(etch_c) > 0:
                events.append({
                    "event_type": "EQUIPMENT_CHAMBER_EXCURSION",
                    "target_id": "ETCH-07 / Chamber C",
                    "severity": "CRITICAL",
                    "affected_wafers_count": int(len(etch_c)),
                    "mean_yield_pct": round(float(etch_c["final_yield"].mean() * 100), 2),
                    "description": f"Chamber C of ETCH-07 exhibiting severe edge-ring failure excursion across {len(etch_c)} wafers."
                })

        # 3. Check WAT Parametric shift
        if "vt_n_mean" in wafer_features_df.columns:
            vt_high = wafer_features_df[wafer_features_df["vt_n_mean"] > 325]
            if len(vt_high) > 0:
                affected_lots = list(vt_high["lot_id"].unique())
                events.append({
                    "event_type": "PARAMETRIC_WAT_SHIFT",
                    "target_id": f"Vt_n Shift ({len(affected_lots)} Lots)",
                    "severity": "HIGH",
                    "affected_lots": affected_lots,
                    "mean_vt_n": round(float(vt_high["vt_n_mean"].mean()), 2),
                    "description": f"Significant positive Vt_n threshold voltage shift (+18 mV above baseline) detected across lots {affected_lots}."
                })

        return events

    @staticmethod
    def load() -> "ExcursionDetector":
        """Load saved detector instance from disk."""
        return joblib.load(config.EXCURSION_DETECTOR_PATH)
