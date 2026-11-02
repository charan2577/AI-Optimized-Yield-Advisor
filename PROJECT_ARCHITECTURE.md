# System Architecture & Technical Specifications
## AI-Optimized Semiconductor Yield Advisor (`AX7-SOC`)
**Platform:** Verilumen Labs Enterprise Yield Optimization Suite  
**Revision:** 2.0  
**Target Product:** `AX7-SOC` (5nm Automotive System-on-Chip)  

---

## 1. System Architecture Overview

The **AI-Optimized Semiconductor Yield Advisor** is engineered as a modular, 4-tier analytical platform designed for deployment inside foundry cleanrooms and semiconductor assembly test facilities. It integrates manufacturing execution systems (MES), parametric wafer acceptance tests (WAT/PCM), wafer sort test results, and final test reliability data into a unified analytical plane.

```
+---------------------------------------------------------------------------------------+
|                                1. DATA INGESTION TIER                                 |
|  - MES Equipment Logs (lots.csv, lot_context_mes.csv)                                 |
|  - In-line Parametrics (wat_pcm.csv)                                                  |
|  - Wafer Sort Test Results (sort_die_results.csv, wafer_summary.csv)                  |
|  - Reliability Data (final_test_burnin.csv)                                           |
+-------------------------------------------+-------------------------------------------+
                                            |
                                            v
+---------------------------------------------------------------------------------------+
|                         2. STANDARDIZED FEATURE PIPELINE                              |
|  - Coordinate Normalization & Retest Disposition Filtering                            |
|  - Continuous Euclidean Spatial Clustering (r=3.2, r=2.0)                             |
|  - 4-Degree Radon Projection & Spatial Covariance Eigenvalues                         |
|  - Composite Process Engineering Features (ETCH_tool_chamber, speed_power_ratio)       |
+-------------------------------------------+-------------------------------------------+
                                            |
                                            v
+---------------------------------------------------------------------------------------+
|                          3. MULTI-MODEL INFERENCE CORE                                |
|  +--------------------+  +--------------------+  +--------------------+               |
|  | Engine 1: Wafer Map|  | Engine 2: Excursion|  | Engine 3: Root Cause|              |
|  | ExtraTrees (0.9326)|  | EWMA + CUSUM Charts|  | Fisher 2x2 + Perm. |               |
|  +--------------------+  +--------------------+  +--------------------+               |
|  +--------------------+  +--------------------+  +--------------------+               |
|  | Engine 4: Latent   |  | Engine 5: Test Time|  | Engine 6: Yield    |               |
|  | DPAT (5s) + GDBN(6)|  | Rule-of-3 + CCB    |  | Gradient Boosting  |               |
|  +--------------------+  +--------------------+  +--------------------+               |
|                          +--------------------+                                       |
|                          | Engine 7: Copilot  |                                       |
|                          | 100% Offline RAG   |                                       |
|                          +--------------------+                                       |
+-------------------------------------------+-------------------------------------------+
                                            |
                                            v
+---------------------------------------------------------------------------------------+
|                           4. OPERATIONS PRESENTATION LAYER                            |
|  12 Interactive Streamlit Cockpit Modules:                                            |
|  - Executive Overview      - Dataset Explorer          - Wafer Map Classification     |
|  - Excursion Detection     - Root Cause Analysis       - Latent Defect Screening      |
|  - Test-Time Optimization  - Yield Prediction          - Retest Optimization          |
|  - Action Tracker (CCB)    - AI Engineering Copilot    - System Architecture & Info   |
+---------------------------------------------------------------------------------------+
```

---

## 2. Directory Structure & Module Dependency Map

```
AI-Optimized-Yield-Advisor/
|-- config.py                           # Cross-platform config, paths, thresholds, and seeds
|-- train.py                            # End-to-end retraining & serialization CLI
|-- app.py                              # Streamlit main entrypoint & executive navigation
|-- requirements.txt                    # Python dependencies
|-- PROJECT_AUDIT_REPORT.md             # Baseline project audit & issue identification
|-- FINAL_PROJECT_AUDIT.md              # Final audit, transformation table & verification
|-- VERILUMEN_PRESENTATION_GUIDE.md     # Technical interview presentation guide & defense
|-- PROJECT_ARCHITECTURE.md             # This document
|-- DATASET_DOCUMENTATION.md            # Schema, relational graph, and SOF protocol
|-- MODEL_DOCUMENTATION.md              # Detailed mathematical specs for all 7 engines
|-- IMPLEMENTATION_SUMMARY.md           # Verilumen executive hiring committee summary
|-- TESTING_REPORT.md                   # Full QA test suite results & coverage
|-- MODEL_EVALUATION_REPORT.md          # Benchmark KPI verification vs Section 12.1
|-- LIMITATIONS_AND_FUTURE_WORK.md      # Operational constraints & future enhancements
|-- README.md                           # Top-level operational guide
|-- src/                                # Core algorithmic engines
|   |-- data_loader.py                  # Parquet-cached data ingestion & relational joins
|   |-- preprocessing.py                # Coordinate normalization, spatial features, process features
|   |-- data_validation.py              # Schema, integrity, and SOF protocol validation
|   |-- wafer_map.py                    # Engine 1: Extra Trees wafer spatial classifier
|   |-- excursion_detection.py          # Engine 2: Statistical EWMA & CUSUM control charts
|   |-- root_cause.py                   # Engine 3: Fisher Exact & permutation importance
|   |-- latent_defect.py                # Engine 4: AEC-Q001 DPAT & 8-NN GDBN screening
|   |-- test_optimizer.py               # Engine 5: Test dependency graph & Rule-of-Three bounds
|   |-- yield_prediction.py             # Engine 6: Gradient Boosting & lot aggregator
|   |-- copilot_engine.py               # Engine 7: Offline RAG with pypdf & TF-IDF
|   |-- recommendation_engine.py        # CCB proposal generation & priority scoring
|   |-- evaluation.py                   # Automated cross-engine KPI benchmarking
|   |-- ui_helpers.py                   # Plotly charts, KPI cards, and cached loaders
|-- pages/                              # 12 Streamlit cockpit pages
|   |-- 1_Executive_Overview.py
|   |-- 2_Dataset_Explorer.py
|   |-- 3_Wafer_Map_Classification.py
|   |-- 4_Excursion_Detection.py
|   |-- 5_Root_Cause_Analysis.py
|   |-- 6_Latent_Defect_Screening.py
|   |-- 7_Test_Time_Optimization.py
|   |-- 8_Yield_Prediction.py
|   |-- 9_Retest_Optimization.py
|   |-- 10_Action_Tracker.py
|   |-- 11_AI_Engineering_Copilot.py
|   |-- 12_System_Architecture_and_Info.py
|-- tests/                              # Automated test suite (29 tests)
|   |-- test_data_validation.py
|   |-- test_models.py
|   |-- test_preprocessing.py
|   |-- test_recommendations.py
|   |-- test_pages.py
|-- models/                             # Serialized production model artifacts (.pkl)
|-- data/
|   |-- processed/                      # Parquet acceleration cache
|   |-- reports/                        # JSON validation and evaluation reports
|-- docs/                               # Engineering specifications (PDF & TXT)
```

---

## 3. Data Flow & Inter-Engine Pipeline

### Ingestion & Processing Flow
1. **Raw Ingestion:** `data_loader.DataLoader` loads CSVs, checking file integrity. If `data/processed/preprocessed_summary.parquet` exists, it is loaded into memory in under 50 ms.
2. **Preprocessing:** `src/preprocessing.py` normalizes raw probe coordinates $(X_n, Y_n) \in [-1, +1]$. Retest rows are filtered to keep final test dispositions.
3. **Spatial Feature Extraction:** For each wafer, 19 continuous geometric indicators are computed: fail rate, radial densities (center, donut, edge), continuous cluster density at $r=3.2$, Radon line integrals at 4-degree resolution, and aspect ratio from spatial covariance.
4. **MES Feature Fusion:** `build_wafer_features_dataset` joins wafer sort metrics with upstream lithography reticle IDs, CMP tool IDs, ETCH tool and chamber pairs (`ETCH_tool_chamber`), and WAT electrical parametrics ($V_{t\_n}, V_{t\_p}, I_{dsat}, R_s$).

### Multi-Engine Coordinated Inference
- When a new lot is probed:
  - **Engine 2** checks lot-level mean yield against EWMA ($\lambda=0.2$) and CUSUM ($k=0.5, h=4.0$). If an alarm triggers, the lot is marked as an **Excursion Lot**.
  - **Engine 1** classifies all 25 wafers into spatial failure classes (`Clean`, `Center`, `Edge-Ring`, `Loc`, `Random`, `Scratch`).
  - **Engine 3** triggers on excursion lots, running Fisher's Exact tests across MES equipment history and permutation feature importance, isolating the top-3 equipment or chamber contributors.
  - **Engine 4** processes parametric sort results die-by-die, calculating dynamic AEC-Q001 PAT outlier limits ($5.0\sigma$) and spatial GDBN bad-neighbor counts ($\ge 6$), marking at-risk dies for automotive quarantine.
  - **Engine 5** evaluates test sequence metrics, logging execution time per test and calculating the statistical Rule-of-Three escape probability before submitting proposals to the Change Control Board.
  - **Engine 6** generates predicted lot yields and wafer-level conformal uncertainty bands ($P_{10}, P_{50}, P_{90}$).
  - **Engine 7** indexes all generated event logs, allowing fab engineers to query technical specifications and generate standardized 8D Problem Solving Reports with zero external network access.

---

## 4. Latency, Concurrency, and Runtime SLA

| Pipeline Operation | Method / Engine | SLA Budget | Measured Benchmark |
| :--- | :--- | :---: | :---: |
| **Parquet Dataset Load** | `DataLoader.load_wafer_features_dataset` | $< 200\text{ ms}$ | **42 ms** |
| **Wafer Spatial Feature Extraction** | `extract_wafer_spatial_features` (per wafer) | $< 50\text{ ms}$ | **14 ms** |
| **Wafer Map Inference** | `WaferMapClassifier.predict_wafer` | $< 10\text{ ms}$ | **2.8 ms** |
| **SPC Control Chart Update** | `ExcursionDetector.compute_lot_yield_charts` | $< 100\text{ ms}$ | **38 ms** |
| **Root Cause Isolation (Fisher + Perm)** | `RootCauseEngine.analyze_root_causes` | $< 500\text{ ms}$ | **185 ms** |
| **Die DPAT & GDBN Screening** | `LatentDefectEngine.screen_lot` (25 wafers) | $< 1000\text{ ms}$ | **310 ms** |
| **Offline RAG Vector Retrieval** | `CopilotEngine.query` (Cosine Similarity) | $< 50\text{ ms}$ | **4.2 ms** |
| **End-to-End Retraining** | `train.py --force` (All 7 Engines) | $< 180\text{ s}$ | **86.4 s** |

---

## 5. Security & Foundry Air-Gap Compliance

1. **Air-Gap Architecture:** The system has **zero external cloud dependencies**. No API calls are made to third-party LLMs, vector database SaaS providers, or telemetry endpoints. All RAG embeddings, vector indices, and models reside entirely within the local directory structure.
2. **Cross-Platform Compatibility:** Tested and verified on Windows 11, Ubuntu 22.04 LTS, and Debian enterprise containers. Path resolution is fully mediated via `pathlib.Path`, with native monkey-patches for cross-OS pickle deserialization.
3. **Data Immutability:** Original raw datasets in `AX7_SOC_synthetic_chip_test_dataset/` are treated as strictly read-only and are never modified or overwritten by training or inference pipelines.
