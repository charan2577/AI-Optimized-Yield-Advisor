# Model Documentation & Algorithmic Specifications
## AI-Optimized Semiconductor Yield Advisor (`AX7-SOC`)
**Platform:** Verilumen Labs Enterprise Yield Optimization Suite  
**Target Device:** `AX7-SOC` (5nm Automotive System-on-Chip)  
**Revision:** 2.0  

---

## 1. Engine 1: Wafer Map Spatial Pattern Classifier

### 1.1 Objective & Architecture
Classifies full-wafer spatial defect topologies into 6 standard SEMI categories: `Clean`, `Center`, `Edge-Ring`, `Loc` (Localized cluster), `Random`, and `Scratch`.

- **Algorithm:** `ExtraTreesClassifier` (Extremely Randomized Trees)
- **Hyperparameters:**
  - `n_estimators`: 200 trees
  - `max_depth`: 12
  - `min_samples_split`: 4
  - `class_weight`: `'balanced'` (Counters severe class imbalance)
  - `random_state`: 2026

### 1.2 Input Features (19 Continuous Spatial Metrics)
1. `fail_rate`: Total wafer defect ratio $\frac{\sum \text{fail}}{\sum \text{gross}}$.
2. `center_density`: Defect density within normalized radius $r \le 0.35$.
3. `donut_density`: Defect density in annular zone $0.35 < r \le 0.75$.
4. `edge_density`: Defect density in periphery zone $r > 0.75$.
5. `radial_gradient`: Linear slope of defect density across radial shells.
6. `center_to_edge_ratio`: $\frac{\text{center\_density} + 10^{-4}}{\text{edge\_density} + 10^{-4}}$.
7. `edge_to_center_ratio`: Inverse of above.
8. `x_mean`, `y_mean`: Spatial centroid coordinates of failing dies.
9. `spatial_std`: Standard deviation of failing die coordinates from centroid.
10. `cluster_density_max`: Peak defect density in sliding spatial window.
11. `reticle_pattern_score`: Modulo periodicity corresponding to litho reticle field step.
12. `angular_variance`: Angular dispersion $\text{Var}(\theta)$ around wafer center.
13. `scratch_line_score`: Peak linear Radon projection across 45 angles ($0^\circ$ to $180^\circ$ in $4^\circ$ steps).
14. `circle_cluster_score`: Continuous Euclidean neighbor count within $r = 3.2$ die pitch.
15. `circle_cluster_ratio`: Proportion of failing dies belonging to dense clusters.
16. `scratch_line_ratio`: Proportion of failing dies aligned along the peak Radon projection angle.
17. `scratch_aspect_ratio`: Eigenvalue ratio $\frac{\lambda_1}{\lambda_2}$ of the failing die spatial covariance matrix.
18. `max_continuous_cluster`: Maximum continuous cluster size at $r = 2.0$ die pitch.
19. `is_retest_wafer`: Retest indicator.

### 1.3 Validation Performance
- **Macro F1-Score:** **0.9326** (Exceeds target $\ge 0.93$)
- **Weighted F1-Score:** **0.9842**
- **Per-Class Metrics:**
  - `Clean`: Precision 99.1%, Recall 99.7%, F1 **0.994** (Support: 788)
  - `Center`: Precision 93.7%, Recall 100.0%, F1 **0.967** (Support: 74)
  - `Edge-Ring`: Precision 100.0%, Recall 100.0%, F1 **1.000** (Support: 15)
  - `Loc`: Precision 93.5%, Recall 82.9%, F1 **0.879** (Support: 35)
  - `Random`: Precision 89.2%, Recall 90.3%, F1 **0.897** (Support: 62)
  - `Scratch`: Precision 83.3%, Recall 88.5%, F1 **0.858** (Support: 26)

---

## 2. Engine 2: Excursion Detection Engine

### 2.1 Mathematical Formulation
Monitors lot-level yield time series using complementary Statistical Process Control (SPC) algorithms:

#### Exponentially Weighted Moving Average (EWMA)
Tracks subtle shifts in process mean ($\sim 0.5\sigma$ to $1.5\sigma$):
$$Z_t = \lambda Y_t + (1 - \lambda) Z_{t-1}, \quad Z_0 = \mu_0$$
$$\text{Control Limits: } \text{UCL/LCL}_t = \mu_0 \pm L \sigma_0 \sqrt{\frac{\lambda}{2 - \lambda} \left[1 - (1 - \lambda)^{2t}\right]}$$
- **Tuned Parameters:** Smoothing factor $\lambda = 0.20$, Limit coefficient $L = 3.00$.

#### Cumulative Sum Control Chart (CUSUM)
Integrates cumulative positive and negative deviations from the in-control target:
$$C_t^+ = \max(0, C_{t-1}^+ + (Y_t - \mu_0) - k\sigma_0)$$
$$C_t^- = \max(0, C_{t-1}^- - (Y_t - \mu_0) - k\sigma_0)$$
- **Tuned Parameters:** Reference allowance $k = 0.50$, Decision threshold $h = 4.00 \sigma_0$.

### 2.2 Operational Performance
- Successfully flagged **22 excursion lots** across the 40-lot history.
- Average detection lead time: **1.8 lots** from excursion onset (well within the 3-lot target).
- Zero false alarms during initial stable baseline production (Lots 0 to 19).

---

## 3. Engine 3: Root Cause Analysis Engine

### 3.1 Statistical Attribution & Permutation Importance
Combines non-parametric statistical hypothesis testing with model-based feature attributions:

1. **Fisher's Exact Test:** Constructed on $2\times 2$ equipment contingency tables comparing excursion lots against baseline lots:
   $$\begin{pmatrix} \text{Target Tool \& Excursion} & \text{Target Tool \& Normal} \\ \text{Other Tools \& Excursion} & \text{Other Tools \& Normal} \end{pmatrix}$$
2. **Permutation Importance:** Evaluated on Random Forest / Gradient Boosting regressors over 10 repeated shuffles of each feature column, measuring yield prediction degradation.

### 3.2 Top-3 Root Cause Findings
- **Rank 1:** Lithography Reticle `RT-118B` ($p = 3.55 \times 10^{-21}$, Yield Impact: $-6.71\text{ pts}$, High Confidence, **Confirmed Physical Root Cause**).
- **Rank 2:** WAT $V_{t\_n}$ Parameter Drift ($p = 1.27 \times 10^{-19}$, $r = -0.2811$, Yield Impact: $-5.35\text{ pts}$, High Confidence, **Confirmed Physical Root Cause**).
- **Rank 3:** ETCH Tool & Chamber `ETCH-07 / Chamber C` ($p = 8.75 \times 10^{-9}$, Yield Impact: $-7.98\text{ pts}$, High Confidence, **Confirmed Physical Root Cause**).
- **Rank 4:** CMP Tool `CMP-01` ($p = 5.90 \times 10^{-8}$, Yield Impact: $-2.29\text{ pts}$, Medium Confidence, *Statistical Equipment Association*).
- **Special Pattern Correlation:** CMP Tool `CMP-03` exhibits an isolated association with the **Center** spatial defect pattern (center defect density 17.5% vs 9.7% baseline, $p < 10^{-7}$) due to polishing pad wear.

---

## 4. Engine 4: Latent Defect Detection (DPAT / GDBN)

### 4.1 Methodology & Standards Compliance
Screening latent reliability defects that pass wafer sort but fail automotive burn-in (AEC-Q100 Grade 1 standards):

1. **Dynamic Part Average Testing (DPAT - AEC-Q001):**
   Calculates robust per-wafer statistical limits on parametric distributions ($I_{DDQ}, V_{DD\_MIN}$):
   $$\text{Robust Median: } \tilde{\mu} = \text{Median}(X), \quad \hat{\sigma}_{\text{robust}} = \frac{\text{IQR}(X)}{1.349}$$
   $$\text{DPAT Limits: } [\tilde{\mu} - k\hat{\sigma}_{\text{robust}}, \quad \tilde{\mu} + k\hat{\sigma}_{\text{robust}}] \quad (k = 5.0\sigma)$$
2. **Good-Die-Bad-Neighborhood (GDBN):**
   Evaluates spatial proximity on the $3\times 3$ die neighborhood (8 adjacent neighbors). Passing dies surrounded by $\ge 6$ failing dies are quarantined.

### 4.2 Calibrated Performance
- **Screened Units:** 1,558 dies flagged out of 446,715 evaluated.
- **Yield Cost:** **0.349 pts** (Meets strict $< 0.350$ pt target).
- **Screening Precision:** **26.57%** (414 true early-life failures out of 1,558 flagged).
- **Burn-in Defect Recall:** **59.14%** (414 of 700 total burn-in fails screened).
- **Field DPPM Reduction:** Dropped from 1,567 DPPM to 642.5 DPPM (**59.0% reduction**).
- **Relative Risk Ratio:** **413.6x** higher failure rate in screened vs unscreened dies.

---

## 5. Engine 5: Test-Time Optimization Engine

### 5.1 Test Dependency & Statistical Escape Bounds
Models the ATE test sequence under the **Stop-on-Fail (SOF)** protocol. Evaluates pairwise failure overlaps, execution times, and escape bounds.

#### Statistical Rule of Three Upper Bound
For tests with $0$ recorded unique failures over $N = 500,000$ tested dies, the 95% confidence upper bound on escape probability is:
$$p_{95\%} = \frac{3}{N} = \frac{3}{500,000} = 6.00 \text{ DPPM}$$

### 5.2 Formulated Change Control Board (CCB) Proposals
- **Proposal `TTO-01` (`RING_OSC` - T4120):**
  - High parametric correlation with `FMAX` ($r = 0.996$). Contributes only 13 unique fails.
  - Action: Move to 5% QA sample audit.
  - Time Saved: $20.9\text{ ms/die}$. Annual Savings: \$16,251.84.
- **Proposal `TTO-02` (`ADC_INL` - T6000):**
  - Zero recorded failures due to upstream filtering; consumes $95.0\text{ ms/die}$.
  - Action: Move to 10% sampling on qualified clean lots.
  - Time Saved: $85.5\text{ ms/die}$. Annual Savings: \$66,484.80.
- **Net Optimization:** **8.65% test time reduction** ($106.4\text{ ms}$ saved per die), yielding **\$82,736.64** annual tester savings under CCB risk gating.

---

## 6. Engine 6: Yield Prediction Engine

### 6.1 Model Architecture & Engineering Features
Predicts wafer sort yield prior to probe using upstream in-line electrical parametrics and MES tool assignments.

- **Regressor:** `GradientBoostingRegressor(n_estimators=100, learning_rate=0.05, max_depth=4)`
- **Key Engineered Features:**
  - `ETCH_tool_chamber`: Composite categorical feature capturing specific chamber drift.
  - `speed_power_ratio`: $\frac{I_{dsat\_n}}{|V_{t\_n}|}$ electrical transconductance ratio.
  - `poly_contact_prod`: $R_{s\_poly} \times R_{\text{contact}}$ interconnect impedance product.
  - `wafer_slot`: Cassette position slot (1 to 25).

### 6.2 Validation Results
- **Lot-Level Chronological Holdout (Lots 30–39):**
  - $R^2 = \mathbf{0.3753}$ (vs baseline model $R^2 = -0.0390$)
  - RMSE = $2.906\text{ pts}$ (vs baseline $3.588\text{ pts}$)
  - MAE = $2.227\text{ pts}$ (Target $< 1.2\text{ pts}$ is **NOT MET**)
- **Lot-Level 5-Fold Cross Validation (Leakage-Free):** MAE = **1.375 pts**.
- **Baseline Comparison:** Outperforms the baseline mean predictor ($3.588\text{ pts}$ MAE) by reducing absolute error by $1.36\text{ pts}$ on holdout lots.
- **Conformal Uncertainty:** Quantile interval prediction ($P_{10}, P_{50}, P_{90}$) with 70.4% empirical coverage.


---

## 7. Engine 7: AI Engineering Copilot (Offline RAG)

### 7.1 Architecture & Grounding
- **Engine:** 100% local, air-gapped Retrieval-Augmented Generation (RAG).
- **Document Ingestion:** `pypdf` extracts text from the 19-page Verilumen Technical Design Specification PDF and repository technical guides.
- **Chunking:** 44 semantic chunks ($100 - 300\text{ words}$) with document and page metadata.
- **Indexing:** TF-IDF n-gram vectorizer with sub-linear term frequency scaling and cosine similarity ranking.
- **Query Latency:** $< 5\text{ ms}$ per query.
- **Output Capabilities:** Returns grounded answers with exact document/page citations, and automatically drafts comprehensive **8D Problem Solving Reports** downloadable as Markdown.
