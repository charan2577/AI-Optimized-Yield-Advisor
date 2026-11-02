"""
Master Training and Pipeline Execution Script.
Trains all AI engines, generates evaluation benchmarks, and persists models.
Usage: python train.py
"""

import sys
import time
import json
import logging
import pandas as pd
import numpy as np

import config
from src.data_loader import DataLoader
from src.data_validation import DataValidator
from src.preprocessing import Preprocessor
from src.wafer_map import WaferMapClassifier
from src.excursion_detection import ExcursionDetector
from src.root_cause import RootCauseEngine
from src.latent_defect import LatentDefectEngine
from src.test_optimizer import TestTimeOptimizer
from src.yield_prediction import YieldPredictor
from src.recommendation_engine import RecommendationEngine
from src.evaluation import ModelEvaluator

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("TrainPipeline")


def main():
    force_recompute = "--force" in sys.argv
    start_time = time.time()
    logger.info("================================================================")
    logger.info("STARTING AI-OPTIMIZED YIELD ADVISOR OFFLINE TRAINING PIPELINE")
    logger.info("Verilumen Labs Pvt. Ltd. - AX7-SOC Chip-Test Benchmark")
    logger.info("================================================================")

    loader = DataLoader()

    # Step 1: Data Validation & Quality Audit
    logger.info("\n--- STEP 1: Running Data Quality Audit and Schema Verification ---")
    validator = DataValidator(loader)
    dq_report = validator.generate_full_data_quality_report()
    logger.info(f"Data quality verdict: {dq_report['verdict']}")
    logger.info(f"Verified {dq_report['summary']['total_lots']} lots, {dq_report['summary']['total_wafers']} wafers, {dq_report['summary']['total_ft_records']:,} final test dies.")

    # Step 2: Preprocessing & Feature Engineering
    logger.info("\n--- STEP 2: Preprocessing & Spatial/Genealogy Feature Engineering ---")
    preprocessor = Preprocessor(loader)
    wafer_features_df = preprocessor.build_wafer_features_dataset(force_recompute=force_recompute)
    retest_analytics = preprocessor.analyze_retest_recovery()
    logger.info(f"Retest recovery rate for contact failures (SB30): {retest_analytics['overall_recovery_rate']*100:.2f}%")

    # Step 3: Engine 1 - Wafer Map Pattern Classification
    logger.info("\n--- STEP 3: Training Engine 1 (Wafer Map Pattern Classifier) ---")
    wafer_gt = loader.load_ground_truth_wafer_labels()
    wafer_classifier = WaferMapClassifier()
    wm_metrics = wafer_classifier.train(wafer_features_df, wafer_gt)
    logger.info(f"Wafer Map Classifier Trained. Macro F1: {wm_metrics['macro_f1']:.4f}, Weighted F1: {wm_metrics['weighted_f1']:.4f}")
    logger.info(f"Top spatial features: {list(wm_metrics['feature_importances'].keys())[:5]}")

    # Step 4: Engine 2 - Excursion Detection (EWMA, CUSUM, Shewhart, Isolation Forest)
    logger.info("\n--- STEP 4: Running Engine 2 (Excursion Detection System) ---")
    lots_df = loader.load_lots()
    wafer_summary_df = loader.load_wafer_summary()
    detector = ExcursionDetector()
    lot_charts = detector.compute_lot_yield_charts(wafer_summary_df, lots_df)
    wafer_anomalies = detector.fit_multivariate_wafer_model(wafer_features_df)
    excursion_events = detector.identify_excursion_events(lot_charts, wafer_features_df)
    
    excursion_metrics = {
        "total_excursion_lots": int(lot_charts["is_excursion_lot"].sum()),
        "ewma_signals_count": int(lot_charts["ewma_alarm"].sum()),
        "cusum_signals_count": int(lot_charts["cusum_alarm"].sum()),
        "multivariate_anomalies_count": int(wafer_anomalies["is_multivariate_anomaly"].sum()),
        "identified_events_count": len(excursion_events),
    }
    logger.info(f"Excursion Detection complete. Flagged {excursion_metrics['total_excursion_lots']} excursion lots and {excursion_metrics['multivariate_anomalies_count']} anomalous wafers.")

    # Step 5: Engine 3 - Root Cause Analysis (Commonality, Kruskal-Wallis, Attribution)
    logger.info("\n--- STEP 5: Running Engine 3 (Root Cause Analysis Engine) ---")
    rc_engine = RootCauseEngine()
    rc_engine.train_attribution_model(wafer_features_df)
    ranked_causes = rc_engine.generate_ranked_root_causes(wafer_features_df)
    logger.info(f"Identified {len(ranked_causes)} prioritized root causes:")
    for rc in ranked_causes[:3]:
        logger.info(f"  Rank {rc['rank']}: {rc['factor']} (Yield Impact: -{rc['yield_impact_pts']:.2f} pts, Conf: {rc['confidence']})")

    # Step 6: Engine 4 - Adaptive Limits & Latent Defect Screening (DPAT + GDBN)
    logger.info("\n--- STEP 6: Running Engine 4 (Adaptive Limits & Latent Defect Screening) ---")
    sort_df = loader.load_sort_die_results()
    final_test_df = loader.load_final_test_burnin()
    latent_engine = LatentDefectEngine()
    latent_metrics = latent_engine.run_full_latent_screening(sort_df, final_test_df)
    logger.info(f"Latent Screening: Flagged {latent_metrics['flagged_screened_dies']} dies. Burn-in fail rate in flagged: {latent_metrics['burnin_fail_rate_flagged_pct']:.2f}% vs unflagged: {latent_metrics['burnin_fail_rate_unflagged_pct']:.4f}%. Relative Risk: {latent_metrics['relative_risk_ratio']:.1f}x. DPPM reduction: {latent_metrics['dppm_reduction_pct']:.1f}%.")

    # Step 7: Engine 5 - Test Time Optimization
    logger.info("\n--- STEP 7: Running Engine 5 (Test Time Optimization Engine) ---")
    test_optimizer = TestTimeOptimizer(loader)
    test_redundancy_df = test_optimizer.analyze_test_redundancy(sort_df)
    test_proposals = test_optimizer.evaluate_optimization_proposals(test_redundancy_df)
    logger.info(f"Test Time Optimization: Total time reduction: {test_proposals['total_test_time_reduction_ms']:.1f} ms ({test_proposals['test_time_reduction_pct']:.2f}%). Potential Annual Savings: ${test_proposals['total_annual_savings_usd']:,.2f}.")

    # Step 8: Engine 6 - Yield Prediction (Chronological Holdout)
    logger.info("\n--- STEP 8: Training Engine 6 (Yield Prediction Model) ---")
    yield_predictor = YieldPredictor()
    yp_metrics = yield_predictor.train_chronological(wafer_features_df, train_lot_ratio=0.75)
    logger.info(f"Yield Predictor Trained: Chronological Holdout MAE: {yp_metrics['mae_pts']:.3f} points, RMSE: {yp_metrics['rmse_pts']:.3f} points, R2: {yp_metrics['r2_score']:.3f}.")

    # Step 9: Advisory Layer - Recommendation Engine
    logger.info("\n--- STEP 9: Generating Engineering Recommendations ---")
    rec_engine = RecommendationEngine()
    recs = rec_engine.generate_all_recommendations(
        excursion_events=excursion_events,
        root_causes=ranked_causes,
        latent_results=latent_metrics,
        test_proposals=test_proposals,
        retest_results=retest_analytics
    )
    logger.info(f"Generated {len(recs)} prioritized advisory engineering recommendations.")

    # Step 10: Engine 7 - LLM Copilot & Document Indexing
    logger.info("\n--- STEP 10: Indexing Technical Documents for Engineering Copilot ---")
    from src.copilot_engine import CopilotEngine
    copilot = CopilotEngine()
    n_chunks = copilot.build_index(force_rebuild=force_recompute)
    logger.info(f"Copilot RAG Index built: {n_chunks} semantic documentation chunks indexed.")

    # Step 11: Model Benchmarking & Consolidated Evaluation Report
    logger.info("\n--- STEP 11: Compiling Evaluation Benchmarks & KPI Verification ---")
    evaluator = ModelEvaluator()
    full_eval_report = evaluator.compile_full_evaluation_report(
        wafer_map_metrics=wm_metrics,
        excursion_metrics=excursion_metrics,
        root_cause_metrics=ranked_causes,
        latent_defect_metrics=latent_metrics,
        test_optimizer_metrics=test_proposals,
        yield_predictor_metrics=yp_metrics
    )

    elapsed = time.time() - start_time
    logger.info("================================================================")
    logger.info(f"PIPELINE COMPLETED SUCCESSFULLY IN {elapsed:.2f} SECONDS")
    logger.info("Models saved in: models/")
    logger.info("Reports saved in: data/reports/")
    logger.info("================================================================")


if __name__ == "__main__":
    main()
