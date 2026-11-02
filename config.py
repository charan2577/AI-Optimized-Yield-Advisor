"""
Configuration and constants for the AI-Optimized Semiconductor Yield Advisor.
Following the Verilumen Labs Technical Design Specification Rev 1.0.
"""

import os
import pathlib
from pathlib import Path

# Ensure seamless cross-platform compatibility between WindowsPath and PosixPath
if os.name == "nt":
    try:
        pathlib.PosixPath = pathlib.WindowsPath
    except Exception:
        pass
else:
    try:
        pathlib.WindowsPath = pathlib.PosixPath
    except Exception:
        pass

# Base Paths
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "AX7_SOC_synthetic_chip_test_dataset"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
REPORTS_DIR = PROJECT_ROOT / "data" / "reports"
MODELS_DIR = PROJECT_ROOT / "models"
DOCS_DIR = PROJECT_ROOT / "docs"

# Dataset Files
SORT_DIE_RESULTS_CSV = DATA_DIR / "sort_die_results.csv"
WAFER_SUMMARY_CSV = DATA_DIR / "wafer_summary.csv"
LOTS_CSV = DATA_DIR / "lots.csv"
LOT_CONTEXT_MES_CSV = DATA_DIR / "lot_context_mes.csv"
WAT_PCM_CSV = DATA_DIR / "wat_pcm.csv"
TEST_LIMITS_CSV = DATA_DIR / "test_limits.csv"
BIN_DEFINITIONS_CSV = DATA_DIR / "bin_definitions.csv"
FINAL_TEST_BURNIN_CSV = DATA_DIR / "final_test_burnin.csv"

# Ground Truth Files (Strictly for validation/calibration, never for model inputs)
GT_WAFER_LABELS_CSV = DATA_DIR / "ground_truth" / "wafer_labels.csv"
GT_DIE_LABELS_CSV = DATA_DIR / "ground_truth" / "die_labels.csv"
GT_INJECTED_EVENTS_CSV = DATA_DIR / "ground_truth" / "injected_events.csv"
DATASET_STATS_JSON = DATA_DIR / "dataset_stats.json"

# Process & Geometry Parameters
PRODUCT_NAME = "AX7-SOC"
TEST_PROGRAM_REV = "AX7_CP1_R4.2"
FAB_NAME = "FAB2"
DIE_GRID_SIZE = 26
GROSS_DIE_PER_WAFER = 500
PROBER_TEMP_C = 85
TOTAL_LOTS = 40
WAFERS_PER_LOT = 25
TOTAL_WAFERS = 1000

# Parametric Test Items in Execution Order
TEST_NAMES = [
    "CONTACT_RES",
    "IDDQ",
    "LEAKAGE",
    "SCAN_FAIL_COUNT",
    "MBIST_FAIL",
    "PLL_LOCK",
    "VMIN",
    "FMAX",
    "RING_OSC",
    "ADC_INL",
]

# Soft Bin Mappings
SOFT_BIN_NAMES = {
    1: "PASS",
    30: "CONTACT",
    12: "IDDQ",
    21: "LEAKAGE",
    3: "SCAN",
    9: "MBIST",
    15: "PLL_LOCK",
    6: "RING_OSC",
    7: "VMIN",
    5: "FMAX",
    18: "ADC_INL",
}

# Wafer Spatial Pattern Classes
PATTERN_CLASSES = [
    "Clean",
    "Edge-Ring",
    "Center",
    "Scratch",
    "Loc",
    "Reticle-Repeat",
]

# Economic & Cost Model Parameters (from Section 12.2 of Technical Design)
VALUE_PER_GOOD_DIE_USD = 8.00
TESTER_COST_PER_SEC_USD = 0.012
ESCAPE_COST_PER_DEFECT_USD = 150.00  # Conservative automotive/industrial escape cost
MONTHLY_WAFER_VOLUME = 10000

# Statistical & Detection Thresholds
EWMA_LAMBDA = 0.20
EWMA_L = 3.0
CUSUM_K = 0.5
CUSUM_H = 4.5
DPAT_SIGMA_MULTIPLIER = 5.0  # Calibrated for optimal DPPM reduction vs yield cost (<0.35 pts)
GDBN_NEIGHBOR_FAIL_THRESHOLD = 6  # Calibrated 8-connected neighbor threshold
MAHALANOBIS_CHI2_PERCENTILE = 0.997

# Model Persistence Paths
WAFER_PATTERN_MODEL_PATH = MODELS_DIR / "wafer_pattern_model.pkl"
EXCURSION_DETECTOR_PATH = MODELS_DIR / "excursion_detector.pkl"
ROOT_CAUSE_ENGINE_PATH = MODELS_DIR / "root_cause_engine.pkl"
LATENT_DEFECT_ENGINE_PATH = MODELS_DIR / "latent_defect_engine.pkl"
TEST_OPTIMIZER_PATH = MODELS_DIR / "test_optimizer.pkl"
YIELD_PREDICTION_MODEL_PATH = MODELS_DIR / "yield_prediction_model.pkl"
COPILOT_INDEX_PATH = MODELS_DIR / "copilot_index.pkl"
PREPROCESSED_SUMMARY_PATH = PROCESSED_DATA_DIR / "preprocessed_summary.parquet"
