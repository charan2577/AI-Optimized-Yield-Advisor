"""
Preprocessing and Feature Engineering Pipeline.
Constructs normalized spatial features, wafer-level summaries,
retest disposition views, and structured modeling tables.
"""

from pathlib import Path
from typing import Dict, Tuple, Optional
import numpy as np
import pandas as pd

import config


class Preprocessor:
    """End-to-end data preprocessing and feature transformation pipeline."""

    def __init__(self, data_loader=None):
        from src.data_loader import DataLoader
        self.loader = data_loader or DataLoader()

    @staticmethod
    def compute_spatial_coordinates(die_x: np.ndarray, die_y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute normalized radial distance (0 to 1) and polar angle (-pi to +pi)
        from die coordinates on a 26x26 grid.
        """
        cx = cy = (config.DIE_GRID_SIZE - 1) / 2.0  # 12.5
        r_edge = config.DIE_GRID_SIZE / 2.0 - 0.4     # 12.6
        dx = die_x - cx
        dy = die_y - cy
        r_norm = np.hypot(dx, dy) / r_edge
        theta = np.arctan2(dy, dx)
        return r_norm, theta

    def build_final_disposition_dataset(self) -> pd.DataFrame:
        """
        Produce a clean final-disposition die dataset.
        Insertion 2 replaces insertion 1 where retested.
        Adds normalized radius and angle.
        """
        sort_df = self.loader.load_sort_die_results()
        final_df = sort_df.sort_values("insertion").drop_duplicates(
            subset=["lot_id", "wafer_id", "die_x", "die_y"], keep="last"
        ).reset_index(drop=True)

        r_norm, theta = self.compute_spatial_coordinates(final_df["die_x"].values, final_df["die_y"].values)
        final_df["r_norm"] = r_norm
        final_df["theta"] = theta
        return final_df

    def analyze_retest_recovery(self) -> Dict[str, pd.DataFrame]:
        """
        Detailed retest recovery analysis:
        Computes how many dies were retested (insertion=2),
        the recovery rate (SB30 -> PASS), breakdown by site and probe card.
        """
        sort_df = self.loader.load_sort_die_results()
        retests = sort_df[sort_df["insertion"] == 2].copy()
        
        # Merge with insertion 1 to see transition
        ins1 = sort_df[sort_df["insertion"] == 1].copy()
        merged = pd.merge(
            ins1,
            retests[["lot_id", "wafer_id", "die_x", "die_y", "soft_bin", "pass"]],
            on=["lot_id", "wafer_id", "die_x", "die_y"],
            suffixes=("_ins1", "_ins2")
        )

        recovery_by_card = merged.groupby("probe_card_id", observed=True).agg(
            retested_dies=("pass_ins2", "count"),
            recovered_dies=("pass_ins2", "sum"),
            recovery_rate=("pass_ins2", "mean")
        ).reset_index()

        recovery_by_site = merged.groupby(["probe_card_id", "site"], observed=True).agg(
            retested_dies=("pass_ins2", "count"),
            recovered_dies=("pass_ins2", "sum"),
            recovery_rate=("pass_ins2", "mean")
        ).reset_index()

        return {
            "total_retests": len(retests),
            "overall_recovery_rate": float(merged["pass_ins2"].mean()) if len(merged) else 0.0,
            "by_card": recovery_by_card,
            "by_site": recovery_by_site,
        }

    def extract_wafer_spatial_features(self, wafer_dies: pd.DataFrame) -> Dict[str, float]:
        """
        Extract rotation-invariant and geometric spatial features for a single wafer.
        Used by the Wafer Map Pattern AI classifier.
        """
        # Ensure we use insertion 1 for pure wafer map signature recognition
        dies = wafer_dies[wafer_dies["insertion"] == 1].copy() if "insertion" in wafer_dies.columns else wafer_dies.copy()
        
        fails = dies[dies["pass"] == 0]
        n_dies = len(dies)
        n_fails = len(fails)
        fail_rate = n_fails / n_dies if n_dies > 0 else 0.0

        if n_fails == 0:
            return {
                "fail_rate": 0.0,
                "center_density": 0.0,
                "donut_density": 0.0,
                "edge_density": 0.0,
                "radial_gradient": 0.0,
                "center_to_edge_ratio": 0.0,
                "edge_to_center_ratio": 0.0,
                "x_mean": 0.0,
                "y_mean": 0.0,
                "spatial_std": 0.0,
                "cluster_density_max": 0.0,
                "reticle_pattern_score": 0.0,
                "angular_variance": 0.0,
                "scratch_line_score": 0.0,
                "circle_cluster_score": 0.0,
                "circle_cluster_ratio": 0.0,
                "scratch_line_ratio": 0.0,
                "scratch_aspect_ratio": 1.0,
                "max_continuous_cluster": 0.0,
            }

        r, theta = self.compute_spatial_coordinates(dies["die_x"].values, dies["die_y"].values)
        dies["r_norm"] = r
        dies["theta"] = theta
        dies["is_fail"] = (dies["pass"] == 0).astype(int)

        # Concentric zones:
        # Center: r < 0.35
        # Donut/Mid: 0.35 <= r < 0.75
        # Edge: r >= 0.75
        center_mask = dies["r_norm"] < 0.35
        donut_mask = (dies["r_norm"] >= 0.35) & (dies["r_norm"] < 0.75)
        edge_mask = dies["r_norm"] >= 0.75

        center_density = dies.loc[center_mask, "is_fail"].mean() if center_mask.sum() > 0 else 0.0
        donut_density = dies.loc[donut_mask, "is_fail"].mean() if donut_mask.sum() > 0 else 0.0
        edge_density = dies.loc[edge_mask, "is_fail"].mean() if edge_mask.sum() > 0 else 0.0

        # Radial gradient & ratio
        radial_gradient = edge_density - center_density
        center_to_edge_ratio = (center_density + 1e-4) / (edge_density + 1e-4)
        edge_to_center_ratio = (edge_density + 1e-4) / (center_density + 1e-4)

        # Centroid of failing dies
        fail_dx = (fails["die_x"] - 12.5).values
        fail_dy = (fails["die_y"] - 12.5).values
        x_mean = float(np.mean(fail_dx))
        y_mean = float(np.mean(fail_dy))
        spatial_std = float(np.std(np.hypot(fail_dx, fail_dy)))

        # Reticle repeating pattern score:
        # Exposure field is 3x3 dies. Reticle defect repeats at specific (die_x % 3, die_y % 3)
        dies["ret_pos"] = (dies["die_x"] % 3) * 3 + (dies["die_y"] % 3)
        reticle_fail_rates = dies.groupby("ret_pos")["is_fail"].mean()
        reticle_max = reticle_fail_rates.max()
        reticle_mean = reticle_fail_rates.mean()
        reticle_pattern_score = float(reticle_max - reticle_mean)

        fx = fails["die_x"].values
        fy = fails["die_y"].values

        # Continuous circle cluster score centered on each failing die
        c_counts = [int(np.sum((fx - xi)**2 + (fy - yi)**2 <= 3.2**2)) for xi, yi in zip(fx, fy)]
        circle_cluster_score = float(max(c_counts)) if c_counts else 0.0
        circle_cluster_ratio = float(circle_cluster_score / (n_fails + 1e-4))

        # Continuous small-radius cluster score (r=2.0)
        c_counts_small = [int(np.sum((fx - xi)**2 + (fy - yi)**2 <= 2.0**2)) for xi, yi in zip(fx, fy)]
        max_continuous_cluster = float(max(c_counts_small)) if c_counts_small else 0.0

        # High-resolution Radon directional projection (4-degree steps, 32 bins)
        max_line_count = 0
        for deg in range(0, 180, 4):
            rad = np.deg2rad(deg)
            proj = fail_dx * np.sin(rad) - fail_dy * np.cos(rad)
            counts, _ = np.histogram(proj, bins=32, range=(-13, 13))
            c_max = counts.max() if len(counts) > 0 else 0
            if c_max > max_line_count:
                max_line_count = c_max
        scratch_line_score = float(max_line_count)
        scratch_line_ratio = float(scratch_line_score / (n_fails + 1e-4))

        # Spatial PCA aspect ratio (elongation of defects)
        if n_fails >= 4:
            try:
                cov = np.cov(fail_dx, fail_dy)
                eigvals = np.linalg.eigvalsh(cov)
                scratch_aspect_ratio = float(np.sqrt(max(eigvals[-1], 1e-4) / max(eigvals[0], 1e-4)))
            except Exception:
                scratch_aspect_ratio = 1.0
        else:
            scratch_aspect_ratio = 1.0

        # 5x5 local density maximum
        grid = np.zeros((config.DIE_GRID_SIZE, config.DIE_GRID_SIZE))
        for _, row in fails.iterrows():
            grid[int(row["die_y"]), int(row["die_x"])] = 1
            
        max_loc_sum = 0
        for i in range(config.DIE_GRID_SIZE - 4):
            for j in range(config.DIE_GRID_SIZE - 4):
                loc_sum = np.sum(grid[i:i+5, j:j+5])
                if loc_sum > max_loc_sum:
                    max_loc_sum = loc_sum
        cluster_density_max = float(max(max_loc_sum / 25.0, circle_cluster_score / 25.0))

        # Angular distribution variance
        fails_theta = dies.loc[dies["is_fail"] == 1, "theta"]
        angular_counts, _ = np.histogram(fails_theta, bins=8, range=(-np.pi, np.pi))
        angular_variance = float(np.std(angular_counts) / (np.mean(angular_counts) + 1e-4))

        return {
            "fail_rate": float(fail_rate),
            "center_density": float(center_density),
            "donut_density": float(donut_density),
            "edge_density": float(edge_density),
            "radial_gradient": float(radial_gradient),
            "center_to_edge_ratio": float(center_to_edge_ratio),
            "edge_to_center_ratio": float(edge_to_center_ratio),
            "x_mean": x_mean,
            "y_mean": y_mean,
            "spatial_std": spatial_std,
            "cluster_density_max": cluster_density_max,
            "reticle_pattern_score": reticle_pattern_score,
            "angular_variance": angular_variance,
            "scratch_line_score": scratch_line_score,
            "circle_cluster_score": circle_cluster_score,
            "circle_cluster_ratio": circle_cluster_ratio,
            "scratch_line_ratio": scratch_line_ratio,
            "scratch_aspect_ratio": scratch_aspect_ratio,
            "max_continuous_cluster": max_continuous_cluster,
        }

    def build_wafer_features_dataset(self, force_recompute: bool = False) -> pd.DataFrame:
        """
        Build complete wafer-level feature dataset:
        Combines spatial features, WAT/PCM summary, MES equipment history,
        and yield targets.
        """
        if not force_recompute and config.PREPROCESSED_SUMMARY_PATH.exists():
            print(f"Loading cached preprocessed wafer summary from {config.PREPROCESSED_SUMMARY_PATH}...")
            return pd.read_parquet(config.PREPROCESSED_SUMMARY_PATH)

        ws = self.loader.load_wafer_summary()
        mes = self.loader.load_lot_context_mes()
        wat = self.loader.load_wat_pcm()
        sort_df = self.loader.load_sort_die_results()

        # 1. Aggregate WAT per wafer (mean across the 5 sites)
        wat_agg = wat.groupby(["lot_id", "wafer_id"]).agg(
            vt_n_mean=("vt_n_mV", "mean"),
            vt_p_mean=("vt_p_mV", "mean"),
            idsat_n_mean=("idsat_n_uA_um", "mean"),
            rs_poly_mean=("rs_poly_ohm_sq", "mean"),
            contact_chain_mean=("contact_chain_ohm", "mean")
        ).reset_index()

        # 2. Pivot key MES steps
        # Steps: ETCH, CMP, LITHO, WAFER_SORT_LOAD, DEP, IMPLANT, ANNEAL, WET_CLEAN
        mes_pivot = mes.pivot_table(
            index=["lot_id", "wafer_id"],
            columns="step",
            values=["equipment_id", "chamber", "reticle_id"],
            aggfunc="first"
        )
        # Flatten multiindex columns
        mes_pivot.columns = [f"{c[1]}_{c[0]}" for c in mes_pivot.columns]
        mes_pivot = mes_pivot.reset_index()

        # 3. Compute spatial features per wafer
        print("Extracting spatial features across 1,000 wafers...")
        spatial_records = []
        for (lot_id, wafer_id), group in sort_df.groupby(["lot_id", "wafer_id"], observed=True):
            feat = self.extract_wafer_spatial_features(group)
            feat["lot_id"] = lot_id
            feat["wafer_id"] = wafer_id
            spatial_records.append(feat)
        spatial_df = pd.DataFrame(spatial_records)

        # 4. Merge all together
        lots = self.loader.load_lots()
        merged = ws.merge(lots[["lot_id", "tester_id", "probe_card_id"]], on="lot_id", how="left")
        merged = merged.merge(wat_agg, on=["lot_id", "wafer_id"], how="left")
        merged = merged.merge(mes_pivot, on=["lot_id", "wafer_id"], how="left")
        merged = merged.merge(spatial_df, on=["lot_id", "wafer_id"], how="left")

        # 5. Composite semiconductor process features
        if "ETCH_equipment_id" in merged.columns and "ETCH_chamber" in merged.columns:
            merged["ETCH_tool_chamber"] = merged["ETCH_equipment_id"].astype(str) + "_" + merged["ETCH_chamber"].astype(str)
        if "idsat_n_mean" in merged.columns and "vt_n_mean" in merged.columns:
            merged["speed_power_ratio"] = merged["idsat_n_mean"] / (merged["vt_n_mean"] + 1e-4)
        if "rs_poly_mean" in merged.columns and "contact_chain_mean" in merged.columns:
            merged["poly_contact_prod"] = merged["rs_poly_mean"] * merged["contact_chain_mean"]
        merged["wafer_slot"] = merged["wafer_id"].astype(float)

        # Save processed dataset
        config.PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
        merged.to_parquet(config.PREPROCESSED_SUMMARY_PATH, index=False)
        print(f"Wafer features dataset built: {merged.shape[0]} wafers, {merged.shape[1]} features.")
        return merged
