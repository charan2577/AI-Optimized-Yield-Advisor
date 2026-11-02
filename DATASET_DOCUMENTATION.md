# Dataset Documentation & Relational Specifications
## AI-Optimized Semiconductor Yield Advisor (`AX7-SOC`)
**Product:** `AX7-SOC` (5nm Automotive System-on-Chip)  
**Dataset Version:** Synthetic Chip-Test Benchmark Rev 1.0  
**Fab / Facility:** FAB2, Cleanroom 5A  
**Total Production Volume:** 40 Lots, 1,000 Wafers, 500,000 Dies  

---

## 1. Relational Database Schema & Data Dictionary

The `AX7-SOC` dataset mirrors standard semiconductor manufacturing execution systems (MES), wafer acceptance testing (WAT/PCM), wafer probe (CP1/Sort), and final assembly test (FT/Burn-in).

```
[lots.csv] (40 Lots)
   |
   +--- 1:N ---> [lot_context_mes.csv] (Equipment, Reticles, Chambers)
   |
   +--- 1:N ---> [wafer_summary.csv] (1,000 Wafers - 25 wafers/lot)
   |                |
   |                +--- 1:N ---> [wat_pcm.csv] (In-line Electrical Parametrics)
   |                |
   |                +--- 1:N ---> [sort_die_results.csv] (500,000 Dies - Die X, Y, Bins)
   |                                 |
   |                                 +--- 1:1 ---> [final_test_burnin.csv] (Post-packaging Burn-In)
   |
   +--- N:1 ---> [test_limits.csv] & [bin_definitions.csv] (ATE Hardware Specifications)
```

---

## 2. Table Specifications & Column Dictionaries

### 2.1 Lots Master Table (`lots.csv`)
Records lot-level manufacturing metadata, tool assignments, and test program revisions.
- **Row Count:** 40 lots (Chronological sequence: `L262200` to `L262239`).
- **Columns:**
  - `lot_id` (`str`, Primary Key): Unique alphanumeric lot identifier.
  - `product` (`str`): Target semiconductor device (`AX7-SOC`).
  - `fab` (`str`): Manufacturing cleanroom facility (`FAB2`).
  - `technology` (`str`): Silicon process node (`5nm`).
  - `wafer_count` (`int`): Uniformly 25 wafers per cassette.
  - `fab_start_ts` (`datetime`): Front-End-of-Line (FEOL) process release timestamp.
  - `sort_start_ts` (`datetime`): Automated Test Equipment (ATE) probe start timestamp.
  - `tester_id` (`str`): Teradyne UltraFLEX-compatible ATE tester identifier (`T01`, `T02`, `T03`, `T04`).
  - `probe_card_id` (`str`): Multi-die probe card serial number (`PC-4410`, `PC-4412`, `PC-4415`).
  - `program_rev` (`str`): ATE test program software revision (`AX7_CP1_R4.2`).
  - `prober_temp_c` (`float`): Controlled chuck temperature in Celsius ($30.0^\circ\text{C}$).

---

### 2.2 MES Equipment Context Table (`lot_context_mes.csv`)
Tracks equipment routing and chamber assignments across FEOL and BEOL processes.
- **Columns:**
  - `lot_id` (`str`, Foreign Key): Links to `lots.csv`.
  - `process_step` (`str`): Semiconductor fabrication process (`LITHO`, `ETCH`, `CMP`, `IMPLANT`, `CVD`, `WAFER_SORT_LOAD`).
  - `equipment_id` (`str`): Primary tool identifier (e.g., `ETCH-07`, `LITHO-02`, `CMP-01`).
  - `chamber` (`str`, Optional): Sub-chamber designation (`A`, `B`, `C`, `D`).
  - `reticle_id` (`str`, Optional): Lithography photomask reticle serial (e.g., `RT-118A`, `RT-118B`).
  - `operator_id` (`str`): Cleanroom technician ID.
  - `step_start_ts` & `step_end_ts` (`datetime`): Processing window timestamps.

---

### 2.3 Wafer Summary Table (`wafer_summary.csv`)
Aggregated wafer-level probe metrics and process dispositions.
- **Row Count:** 1,000 wafers (Exactly 25 wafers per lot).
- **Columns:**
  - `lot_id` (`str`, Foreign Key): Associated lot.
  - `wafer_id` (`int`, 1 to 25): Physical slot number within standard FOUP cassette.
  - `gross_die` (`int`): Uniformly 500 dies per 300mm wafer.
  - `pass_die` (`int`): Count of electrically functional dies (Bin 1).
  - `fail_die` (`int`): Count of defective dies (Bins 2 through 12).
  - `yield_pct` (`float`): Final sort yield percentage: $\frac{\text{pass\_die}}{\text{gross\_die}} \times 100\%$.
  - `mean_test_time_ms` (`float`): Average tester probe dwell time per die.

---

### 2.4 Wafer Acceptance Test / PCM Table (`wat_pcm.csv`)
Electrical parametric test data measured on scribe-line test structures across 5 site locations per wafer (Center, North, South, East, West).
- **Key Electrical Parametrics:**
  - `vt_n_mean` ($V_{t\_n}$): NMOS saturation threshold voltage (mV). Nominal: $310\text{ mV}$.
  - `vt_p_mean` ($V_{t\_p}$): PMOS saturation threshold voltage (mV). Nominal: $-295\text{ mV}$.
  - `idsat_n_mean` ($I_{dsat\_n}$): NMOS saturation drive current ($\mu\text{A}/\mu\text{m}$). Nominal: $1080\text{ }\mu\text{A}/\mu\text{m}$.
  - `rs_poly_mean` ($R_{s\_poly}$): Polysilicon sheet resistance ($\Omega/\Box$). Nominal: $7.8\text{ }\Omega/\Box$.
  - `contact_chain_mean`: Scribe-line contact via chain resistance ($\Omega$). Nominal: $42\text{ }\Omega$.
  - `eot_mean`: Equivalent Oxide Thickness ($\text{nm}$). Nominal: $1.15\text{ nm}$.

---

### 2.5 Wafer Sort Die Results Table (`sort_die_results.csv`)
High-granularity, die-level test records across all 500,000 tested dies.
- **Row Count:** 500,000 records.
- **Columns:**
  - `lot_id` (`str`) & `wafer_id` (`int`): Wafer identification.
  - `die_x` (`int`, $-12$ to $+13$) & `die_y` (`int`, $-12$ to $+13$): Die grid coordinates on 300mm circular grid ($26\times 26$ bounding envelope).
  - `hard_bin` (`int`): Hardware bin assignment:
    - `Bin 1`: Functional Pass.
    - `Bin 2`: Continuity / Contact Open/Short Fail.
    - `Bin 3`: Leakage / Gross $I_{DDQ}$ Fail.
    - `Bin 4`: Digital Core Logic / BIST Fail.
    - `Bin 5`: SRAM Memory Repairable/Unrepairable Fail.
    - `Bin 6`: High-Speed SerDes / PHY Fail.
    - `Bin 7`: Analog / ADC Non-Linearity Fail.
    - `Bin 8`: Thermal / Clock Frequency Fail.
  - `soft_bin` (`int`): Granular diagnostic test bin (1 to 24).
  - `is_pass` (`bool`): Functional status (`True` if Hard Bin == 1, else `False`).
  - `is_retest` (`bool`): Flag indicating if the die underwent immediate prober retest.
  - `first_fail_test_id` (`str`): Test ID that triggered the stop-on-fail abort.
  - `test_time_ms` (`float`): Total dwell time consumed testing this die.

---

### 2.6 Final Test & Burn-In Reliability Table (`final_test_burnin.csv`)
Tracks packaged dies through post-assembly High-Temperature Operating Life (HTOL) burn-in testing.
- **Row Count:** 446,715 packaged dies (Only Bin 1 sort passing dies proceed to assembly).
- **Columns:**
  - `lot_id`, `wafer_id`, `die_x`, `die_y`: Traceability coordinates linking back to sort die.
  - `burnin_failed` (`bool`): `True` if the die experienced an early life failure during 168-hour HTOL burn-in. Total burn-in failures: **700 dies** (Baseline early life failure rate: 1,567 DPPM).

---

## 3. Stop-on-Fail (SOF) Testing Protocol

Automated Test Equipment (ATE) operates under a strict **Stop-on-Fail (SOF)** protocol to minimize overall test cost:
1. Tests execute in a predefined linear execution sequence (Tests $T_{1000} \rightarrow T_{1010} \rightarrow \dots \rightarrow T_{6000}$).
2. The instant a die violates any test limit, the tester immediately flags the failure, assigns the corresponding Hard/Soft Bin, and **aborts all subsequent tests for that die**.
3. Therefore, downstream tests (e.g., `ADC_INL` at $T_{6000}$) are only executed on dies that have passed all preceding continuity, leakage, logic, and memory tests.
4. **Data Integrity Implication:** The fact that a downstream test records zero failures in `sort_die_results.csv` does **not** indicate that the test is redundant; it indicates that defective units were filtered out upstream.

---

## 4. Data Quality & Integrity Validation Results

The data validation pipeline (`src/data_validation.py`) executes automated checks on ingestion:
- **Schema Validation:** 100% compliance across all 8 relational tables.
- **Foreign Key Integrity:** 100% of wafers in `wafer_summary.csv` match parent lots in `lots.csv`.
- **Wafer Count Uniformity:** Exactly 25 wafers per lot across all 40 lots.
- **Stop-on-Fail Compliance:** 100.0% adherence verified; 0 dies continue testing after a failure.
- **Leakage Prevention:** Chronological 75/25 split partitions Lots 0–29 (Training) and Lots 30–39 (Holdout) with **zero lot sharing or temporal contamination**.
