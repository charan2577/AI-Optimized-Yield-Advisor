# Technical Presentation & Demonstration Guide
### AI-Optimized Yield Advisor for Semiconductor Chip Testing
**Verilumen Labs Pvt. Ltd. | 45-Minute Hiring Assessment & Executive Demonstration**

---

## 1. 45-Minute Demonstration Agenda

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ 00:00 - 05:00 │ Executive Summary, Semiconductor Problem & Financial Impact │
├─────────────────────────────────────────────────────────────────────────────┤
│ 05:00 - 18:00 │ End-to-End Live Streamlit Dashboard Demonstration (10 Pages)│
├─────────────────────────────────────────────────────────────────────────────┤
│ 18:00 - 28:00 │ AI/ML Engineering Deep-Dive (Formulations, SPC, DPAT, Rule-3)│
├─────────────────────────────────────────────────────────────────────────────┤
│ 28:00 - 35:00 │ Clean Codebase Architecture, Integrity Gates & Unit Tests   │
├─────────────────────────────────────────────────────────────────────────────┤
│ 35:00 - 45:00 │ Rigorous Technical Q&A with Senior Interview Panel          │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Phase-by-Phase Walkthrough Script

### Phase 1: Context & Business Stakes (00:00 - 05:00)
- **Opening**: "Good morning panel. Today I am presenting an end-to-end, production-ready AI-Optimized Yield Advisor developed against Verilumen Labs' technical design document and the AX7-SOC 1,000-wafer dataset."
- **The Core Problem**: In a modern high-volume fab running 10,000 wafers/month, traditional weekly yield reviews detect excursions too late. Overkill scraps good dies due to contact wear, while underkill allows latent defect dies to escape to automotive customers.
- **The Value Proposition**: "Our system recovers 1.5 yield points, eliminates 40% of infant-mortality escapes, cuts test time by 8.65%, and unlocks **$8.1 Million in annualized net benefit**."

---

### Phase 2: Live Streamlit Application Tour (05:00 - 18:00)

#### 1. Executive Cockpit (`pages/1_Executive_Dashboard.py`)
- Highlight top KPIs: 40 lots, 1,000 wafers, 501,440 sort tests, 89.34% final yield.
- Show the chronological lot yield trend with baseline at 91.7% and red excursion alarm flags.
- Point out the Soft-Bin Failure Pareto chart showing SCAN, IDDQ, and LEAKAGE as the dominant loss contributors.

#### 2. Data Explorer (`pages/2_Data_Explorer.py`)
- Switch datasets dynamically (Lots, Wafers, Limits, MES, WAT/PCM).
- Emphasize the **Stop-on-Fail audit**: demonstrate that subsequent tests on failing dies are intentionally `NaN`, not corrupted or blindly zero-filled.
- Show the automated Data Quality Report JSON download.

#### 3. Wafer Map Analysis (`pages/3_Wafer_Map_Analysis.py`)
- Select an **Edge-Ring** wafer (e.g., from ETCH-07 Chamber C).
- Toggle between Pass/Fail view and Soft-Bin categorical colormap.
- Show the AI Spatial Classifier predicting **Edge-Ring with >98% confidence**.
- Enable the **Ground-Truth Evaluation Mode toggle** to verify a perfect 1:1 match against `wafer_labels.csv`.

#### 4. Excursion Monitoring (`pages/4_Excursion_Monitoring.py`)
- Walk through the **EWMA control chart ($\lambda=0.20$)** and explain how it catches small $1.5\sigma$ shifts lots earlier than single-point Shewhart charts.
- Show the **CUSUM negative deviation accumulator** crossing threshold $H=4.5\sigma$.
- Review the synthesized incidents (ETCH-07 Chamber C, CMP-03, $V_{t\_n}$ drift).

#### 5. Root Cause Analysis (`pages/5_Root_Cause_Analysis.py`)
- Show the **Statistical Commonality Table**: Fisher exact $p < 10^{-12}$ isolating ETCH-07 Chamber C, and Kruskal-Wallis isolating LITHO Reticle RT-118B.
- Show the **Probe Card Retest tab**: explain that PC-4412 Site 3 suffered needle wear, and that **65.35%** of contact fails recovered on insertion 2.
- Show the **WAT Correlation scatter plot**: explain how frontend $V_{t\_n}$ drift (+18 mV) directly lowered sort yield.

#### 6. Latent Defect Screening (`pages/6_Latent_Defect_Detection.py`)
- Explain the AEC-Q001 **Dynamic Part Average Testing (DPAT)** IDDQ histogram.
- Show the **137.9x Relative Risk ratio**: flagged dies have a 12.94% infant burn-in failure rate compared to 0.0938% in the clean unflagged population.
- Highlight the **40.1% DPPM reduction** achieved at a minor 0.49% yield cost.

#### 7. Test Time Optimization (`pages/7_Test_Time_Optimization.py`)
- Show the correlation matrix proving $r = 0.996$ between `RING_OSC` and `FMAX`.
- Explain the **Rule-of-Three escape bound** ($< 6.0$ DPPM).
- Present the CCB approval proposal to save **106.4 ms per die (8.65%)**, unlocking **$82,736/year** in tester savings.

#### 8. Yield Prediction (`pages/8_Yield_Prediction.py`)
- Show pre-probe forecasting using only upstream WAT and MES data.
- Demonstrate the **Interactive Pre-Probe Yield Simulator**: move the $V_{t\_n}$ slider or select ETCH-07/C to observe projected yield drops in real time.

#### 9. Engineering Recommendations (`pages/9_Engineering_Recommendations.py`)
- Show the prioritized decision inbox (REC-01 to REC-07).
- Demonstrate the **Human-in-the-Loop workflow**: click **Approve** on REC-01, record an engineering note, and observe status transition. Emphasize that Level 1 autonomy ensures compliance with semiconductor change control.

#### 10. Model Performance (`pages/10_Model_Performance.py`)
- Review the consolidated scorecard against Section 12.1 targets.
- Inspect the confusion matrices and model artifact file registry in `models/`.

---

### Phase 3: Technical Formulations Deep-Dive (18:00 - 28:00)

Prepare to walk the panel through key mathematical equations on the whiteboard/slides:
1. **Random Defect vs Systematic**: Negative Binomial $Y_{\text{rand}} = (1 + A D_0 / \alpha)^{-\alpha}$.
2. **EWMA Variance & Control Limits**: $\sigma_{Z_t}^2 = \sigma_0^2 \frac{\lambda}{2-\lambda} [1 - (1-\lambda)^{2t}]$.
3. **Robust DPAT Limits**: $\text{median} \pm 6 \cdot \frac{Q_3 - Q_1}{1.35}$.
4. **Rule of Three 95% Escape Bound**: $p \le \frac{3}{N} \implies \text{DPPM} \le \frac{3}{N} \times 10^6$.
5. **Recommendation Scoring**: $\text{Score} = P(\text{correct}) \cdot \text{Benefit}_{\$} - \text{QualityCost}_{\$} - \text{Effort}_{\$}$.

---

### Phase 4: Code Architecture & Validation (28:00 - 35:00)

- Show the modular repository layout: `src/`, `pages/`, `models/`, `tests/`, `docs/`.
- Run `python -m pytest tests/ -v` live to show **22 passing unit tests** executing in under 10 seconds.
- Highlight **zero target leakage**: prove that `ground_truth/` is never imported in `src/preprocessing.py` or training feature pipelines.

---

### Phase 5: Expected Technical Q&A & Rebuttal Answers (35:00 - 45:00)

#### Q1: "How did you handle missing values in sort_die_results.csv?"
> **Answer**: "In automated chip testing, testers employ a stop-on-fail protocol. When a die fails a test (e.g. IDDQ), subsequent tests (VMIN, FMAX) are never executed. Replacing those missing values with zero or the mean would create a catastrophic artificial mode in the distribution. We maintained `NaN` masks for unexecuted tests, computed robust medians exclusively over executed passing dies, and used stop-on-fail test sequence order as an explicit feature in our test optimizer."

#### Q2: "Why did you choose Random Forest and Gradient Boosting instead of deep neural networks?"
> **Answer**: "While the design document mentions CNNs on $64 \times 64$ images, semiconductor production environments require sub-200 ms inference on standard CPU probers, full explainability, and small memory footprints. Our feature engineering extracts domain-specific rotation-invariant features: concentric radial density gradients for edge rings, Radon line projections for scratches, and modulo-3 field variance for reticle repeats. This allowed an ensemble of Random Forests to achieve a **0.9700 Weighted F1** on CPU in under 5 milliseconds with complete SHAP feature explainability."

#### Q3: "How does your system guarantee that removing a test won't cause catastrophic customer escapes?"
> **Answer**: "We strictly adhere to the statistical **Rule of Three**. When a test exhibits zero unique failures in $N=500,000$ dies, the 95% upper confidence bound on the escape rate is $3/N$, or $6.0$ DPPM. Furthermore, our engine does not propose outright deletion; it recommends transitioning the item to a 5% continuous QA sample audit. If an upstream process shift occurs, the audit sample catches it immediately before significant volume can ship."

#### Q4: "How do you distinguish correlation from causation in root cause analysis?"
> **Answer**: "Process equipment assignment is inherently confounded by time and lot sequencing. Our engine combines classical contingency Fisher exact tests with non-parametric Kruskal-Wallis tests and tree attribution. Crucially, every recommendation is explicitly tagged as a *Statistical Association* and paired with a recommended validation action—such as an MES chamber inhibit pending physical inspection or a controlled split-lot experiment—prior to permanent recipe alterations."
