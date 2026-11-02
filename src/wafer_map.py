"""
ENGINE 1: Wafer Map Pattern Classification Engine.
Extracts spatial failure signatures and classifies wafer maps into
Clean, Edge-Ring, Center, Scratch, Loc (Local Cluster), and Reticle-Repeat.
"""

import joblib
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold

import config


class WaferMapClassifier:
    """Supervised wafer map spatial signature classifier."""

    FEATURE_COLS = [
        "fail_rate",
        "center_density",
        "donut_density",
        "edge_density",
        "radial_gradient",
        "center_to_edge_ratio",
        "edge_to_center_ratio",
        "x_mean",
        "y_mean",
        "spatial_std",
        "cluster_density_max",
        "reticle_pattern_score",
        "angular_variance",
        "scratch_line_score",
        "circle_cluster_score",
        "circle_cluster_ratio",
        "scratch_line_ratio",
        "scratch_aspect_ratio",
        "max_continuous_cluster",
    ]

    def __init__(self):
        self.model = ExtraTreesClassifier(
            n_estimators=200,
            max_depth=12,
            class_weight="balanced",
            random_state=2026,
            n_jobs=-1,
        )
        self.classes_ = None
        self.is_trained = False
        self.evaluation_metrics = {}

    def train(self, features_df: pd.DataFrame, gt_labels_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Train classifier using extracted spatial features and ground truth labels.
        Labels in gt_labels_df: pattern_labels ('Clean', 'Edge-Ring', 'Center', 'Scratch', 'Loc', 'Reticle-Repeat').
        Handles any multi-label string by taking the primary injected pattern.
        """
        # Align on lot_id and wafer_id
        merged = features_df.merge(
            gt_labels_df[["lot_id", "wafer_id", "pattern_labels"]],
            on=["lot_id", "wafer_id"],
            how="inner"
        )

        # Simplify primary pattern
        def parse_primary_label(lbl: str) -> str:
            if not isinstance(lbl, str) or lbl == "" or lbl == "Clean":
                return "Clean"
            # If multiple labels, pick the prominent pattern
            for pat in ["Edge-Ring", "Center", "Scratch", "Reticle-Repeat", "Loc"]:
                if pat in lbl:
                    return pat
            return "Clean"

        merged["target_label"] = merged["pattern_labels"].apply(parse_primary_label)
        
        X = merged[self.FEATURE_COLS].fillna(0)
        y = merged["target_label"]

        # Stratified K-Fold Cross Validation for honest evaluation
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=2026)
        y_true_all, y_pred_all = [], []
        rf_true_all, rf_pred_all = [], []
        gb_true_all, gb_pred_all = [], []

        for train_idx, val_idx in skf.split(X, y):
            X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
            X_va, y_va = X.iloc[val_idx], y.iloc[val_idx]

            # 1. Extra Trees (Primary / Target >= 0.93)
            fold_model = ExtraTreesClassifier(
                n_estimators=200,
                max_depth=12,
                class_weight="balanced",
                random_state=2026,
                n_jobs=-1
            )
            fold_model.fit(X_tr, y_tr)
            y_pred = fold_model.predict(X_va)
            y_true_all.extend(y_va)
            y_pred_all.extend(y_pred)

            # 2. Random Forest benchmark
            rf_fold = RandomForestClassifier(
                n_estimators=150,
                max_depth=10,
                class_weight="balanced_subsample",
                random_state=2026,
                n_jobs=-1
            )
            rf_fold.fit(X_tr, y_tr)
            rf_pred_all.extend(rf_fold.predict(X_va))
            rf_true_all.extend(y_va)

            # 3. Gradient Boosting benchmark
            gb_fold = GradientBoostingClassifier(
                n_estimators=100,
                max_depth=4,
                learning_rate=0.08,
                random_state=2026
            )
            gb_fold.fit(X_tr, y_tr)
            gb_pred_all.extend(gb_fold.predict(X_va))
            gb_true_all.extend(y_va)

        # Train final production model on full dataset
        self.model.fit(X, y)
        self.classes_ = list(self.model.classes_)
        self.is_trained = True

        # Calculate metrics for Extra Trees
        macro_f1 = f1_score(y_true_all, y_pred_all, average="macro")
        weighted_f1 = f1_score(y_true_all, y_pred_all, average="weighted")
        report = classification_report(y_true_all, y_pred_all, output_dict=True)
        cm = confusion_matrix(y_true_all, y_pred_all, labels=self.classes_)

        # Benchmark comparison
        rf_macro = f1_score(rf_true_all, rf_pred_all, average="macro")
        rf_weighted = f1_score(rf_true_all, rf_pred_all, average="weighted")
        gb_macro = f1_score(gb_true_all, gb_pred_all, average="macro")
        gb_weighted = f1_score(gb_true_all, gb_pred_all, average="weighted")

        # Feature importances
        importances = dict(zip(self.FEATURE_COLS, [round(float(v), 4) for v in self.model.feature_importances_]))
        sorted_imp = dict(sorted(importances.items(), key=lambda x: x[1], reverse=True))

        self.evaluation_metrics = {
            "macro_f1": round(float(macro_f1), 4),
            "weighted_f1": round(float(weighted_f1), 4),
            "classification_report": report,
            "confusion_matrix": cm.tolist(),
            "classes": self.classes_,
            "feature_importances": sorted_imp,
            "total_samples": len(y),
            "model_architecture": "ExtraTreesClassifier (200 trees, max_depth=12, balanced)",
            "model_comparison": {
                "Extra Trees (Selected Production)": {
                    "macro_f1": round(float(macro_f1), 4),
                    "weighted_f1": round(float(weighted_f1), 4),
                },
                "Random Forest": {
                    "macro_f1": round(float(rf_macro), 4),
                    "weighted_f1": round(float(rf_weighted), 4),
                },
                "Gradient Boosting": {
                    "macro_f1": round(float(gb_macro), 4),
                    "weighted_f1": round(float(gb_weighted), 4),
                },
            },
        }

        # Save trained model to disk
        config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, config.WAFER_PATTERN_MODEL_PATH)

        return self.evaluation_metrics

    def predict_wafer(self, wafer_features: Any) -> Tuple[str, float, Dict[str, float]]:
        """
        Predict failure pattern for a single wafer given its extracted spatial features.
        Uses a Pandas DataFrame with exact feature names and ordering to prevent scikit-learn warnings.
        Returns: (predicted_pattern, confidence, probability_dict)
        """
        if not self.is_trained:
            raise RuntimeError("WaferMapClassifier must be trained or loaded before prediction.")

        if isinstance(wafer_features, pd.DataFrame):
            input_df = wafer_features[self.FEATURE_COLS]
        elif isinstance(wafer_features, pd.Series):
            input_df = pd.DataFrame([wafer_features[self.FEATURE_COLS].to_dict()], columns=self.FEATURE_COLS)
        else:
            row_dict = {c: [wafer_features.get(c, 0.0)] for c in self.FEATURE_COLS}
            input_df = pd.DataFrame(row_dict, columns=self.FEATURE_COLS)

        probs = self.model.predict_proba(input_df)[0]
        max_idx = np.argmax(probs)
        pred_label = self.classes_[max_idx]
        confidence = float(probs[max_idx])
        prob_dict = {cls: round(float(p), 4) for cls, p in zip(self.classes_, probs)}

        return pred_label, confidence, prob_dict

    @staticmethod
    def load() -> "WaferMapClassifier":
        """Load saved model instance from disk."""
        return joblib.load(config.WAFER_PATTERN_MODEL_PATH)
