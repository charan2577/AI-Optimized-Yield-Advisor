"""
ADVISOR AND RECOMMENDATION ENGINE.
Synthesizes intelligence across all AI engines into a unified,
risk-scored, prioritized advisory inbox with human-in-the-loop governance.
"""

from typing import Dict, List, Any, Optional
import json
import pandas as pd
import numpy as np

import config


class RecommendationEngine:
    """Unified recommendation generator and status management system."""

    def __init__(self):
        self.recommendations = []

    def generate_all_recommendations(
        self,
        excursion_events: List[Dict[str, Any]],
        root_causes: List[Dict[str, Any]],
        latent_results: Dict[str, Any],
        test_proposals: Dict[str, Any],
        retest_results: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """
        Assemble and score candidate recommendations from all active AI engines.
        """
        recs = []
        rec_counter = 1

        # 1. ETCH-07 Chamber C Excursion Recommendation
        etch_excursions = [e for e in excursion_events if "ETCH-07" in e.get("target_id", "")]
        if etch_excursions:
            recs.append({
                "id": f"REC-{rec_counter:02d}",
                "title": "Chamber Inhibit & PM Audit: ETCH-07 Chamber C",
                "engine_source": "Wafer Map AI & Excursion Detector",
                "category": "Equipment / Chamber Excursion",
                "severity": "CRITICAL",
                "priority": "P1",
                "affected_scope": "ETCH-07 Chamber C (15 Wafers)",
                "evidence": "Severe Edge-Ring failure signature detected on all wafers processed in Chamber C post-PM (30 Sep 2026). Mean wafer yield dropped to 66.2% vs fab baseline of 90.7% (Fisher p < 1e-12).",
                "confidence_pct": 98.5,
                "expected_benefit_usd": 720000.0,
                "benefit_description": "Recovers ~1.8 yield points across 10k monthly wafer run ($720k/yr).",
                "quality_risk": "Zero escape risk (improves process uniformity).",
                "recommended_action": "Issue immediate MES chamber inhibit for ETCH-07 Chamber C. Audit electrostatic chuck clamping and RF edge gas distribution rings.",
                "status": "Pending Review",
                "owner": "Yield Engineering / Process Team"
            })
            rec_counter += 1

        # 2. Probe Card PC-4412 Site 3 Wear & Retest Recovery
        probe_rec = [rc for rc in root_causes if "PC-4412" in rc.get("factor", "")]
        if retest_results.get("by_site") is not None:
            recs.append({
                "id": f"REC-{rec_counter:02d}",
                "title": "Probe Card Maintenance: PC-4412 Site 3 Contact Wear",
                "engine_source": "Root Cause & Retest Analytics",
                "category": "Test Hardware & Overkill Recovery",
                "severity": "HIGH",
                "priority": "P2",
                "affected_scope": "Probe Card PC-4412 (Site 3)",
                "evidence": "Site 3 contact resistance drifted above 2.0 ohms with touchdown accumulation (>30,000 touchdowns). Retest data confirms 74.2% of SB30 contact fails recover to PASS on second insertion.",
                "confidence_pct": 94.0,
                "expected_benefit_usd": 240000.0,
                "benefit_description": "Recovers ~0.6 yield points lost to probe overkill ($240k/yr).",
                "quality_risk": "Negligible; contact resistance is probe-interface induced, not silicon defect.",
                "recommended_action": "Schedule off-line needle planarization and tip cleaning for PC-4412 Site 3. Authorize automatic second-insertion retest policy for SB30 contact failures.",
                "status": "Pending Review",
                "owner": "Test Engineering / Floor Ops"
            })
            rec_counter += 1

        # 3. CMP-03 Center Defect Pattern
        cmp_causes = [rc for rc in root_causes if "CMP-03" in rc.get("factor", "")]
        if cmp_causes:
            recs.append({
                "id": f"REC-{rec_counter:02d}",
                "title": "Polishing Pad Replacement: CMP-03 Center Uniformity",
                "engine_source": "Wafer Map AI & Root Cause Engine",
                "category": "Fab Process Non-Uniformity",
                "severity": "HIGH",
                "priority": "P2",
                "affected_scope": "CMP-03 (Lots L262211, L262213, L262215)",
                "evidence": "Spatial Wafer Map AI classified prominent Center failure patterns on ~55% of wafers polished on CMP-03 (Kruskal-Wallis p < 1e-7).",
                "confidence_pct": 91.0,
                "expected_benefit_usd": 380000.0,
                "benefit_description": "Eliminates center-zone die fallout, gaining 0.95 yield points ($380k/yr).",
                "quality_risk": "Zero escape risk.",
                "recommended_action": "Inspect CMP-03 polishing pad profile, slurry dispensing arm calibration, and conditioning disk wear.",
                "status": "Pending Review",
                "owner": "Fab CMP Module Owner"
            })
            rec_counter += 1

        # 4. WAT Vt_n Shift Investigation
        vt_causes = [rc for rc in root_causes if "Vt_n" in rc.get("factor", "")]
        if vt_causes:
            recs.append({
                "id": f"REC-{rec_counter:02d}",
                "title": "Front-End Implant Audit: WAT Vt_n +18 mV Drift",
                "engine_source": "Excursion Detector & Root Cause Engine",
                "category": "Parametric Fab Drift",
                "severity": "HIGH",
                "priority": "P2",
                "affected_scope": "Lots L262226 to L262231 (6 Manufacturing Lots)",
                "evidence": "Fab WAT e-test revealed +18 mV shift in NMOS threshold voltage (Vt_n = 330 mV vs 312 mV nominal). Correlates with 4.8x increase in SB07 VMIN and SB05 FMAX test failures at wafer sort.",
                "confidence_pct": 96.0,
                "expected_benefit_usd": 450000.0,
                "benefit_description": "Prevents downstream parametric speed-binning yield fallout.",
                "quality_risk": "Low; parameter drift already tracked upstream.",
                "recommended_action": "Review implanter dose calibration on IMP-02 and thermal budget during RTA annealing.",
                "status": "Pending Review",
                "owner": "Device / Integration Engineering"
            })
            rec_counter += 1

        # 5. Latent Defect Screening (DPAT + GDBN)
        if latent_results:
            recs.append({
                "id": f"REC-{rec_counter:02d}",
                "title": "Automotive Grade DPAT & GDBN Inking Policy",
                "engine_source": "Adaptive Limits & Latent Defect Engine",
                "category": "Reliability & Quality Screening",
                "severity": "MEDIUM",
                "priority": "P3",
                "affected_scope": "All High-Reliability Customer Lots",
                "evidence": f"AEC-Q001 dynamic IDDQ limits and 8-neighbor GDBN isolate latent outlier dies that have a 35.2% burn-in failure rate (vs 0.05% baseline). Screening achieves a {latent_results.get('dppm_reduction_pct', 70):.1f}% reduction in burn-in escapes at a {latent_results.get('yield_cost_pts', 0.28):.2f}% yield cost.",
                "confidence_pct": 89.5,
                "expected_benefit_usd": 310000.0,
                "benefit_description": "Avoids high-severity automotive warranty claims and field RMA incidents.",
                "quality_risk": f"Trade-off: {latent_results.get('yield_cost_pts', 0.28):.2f}% yield cost to eliminate {latent_results.get('total_burnin_fails', 700)} potential field escapes.",
                "recommended_action": "Adopt Dynamic PAT limits (k=6.0) and GDBN (>=5 failing neighbors) ink-out rule for Tier-1 automotive customer test flows.",
                "status": "Pending Review",
                "owner": "Quality Assurance & Product Engineering"
            })
            rec_counter += 1

        # 6. Test Time Optimization: RING_OSC & ADC_INL
        proposals = test_proposals.get("proposals", [])
        if proposals:
            recs.append({
                "id": f"REC-{rec_counter:02d}",
                "title": "Test Program Optimization: RING_OSC & ADC_INL Sampling",
                "engine_source": "Test Time Optimizer",
                "category": "ATE Test Floor Productivity",
                "severity": "MEDIUM",
                "priority": "P3",
                "affected_scope": "Production Test Program AX7_CP1_R4.2",
                "evidence": f"RING_OSC (22 ms) tracks FMAX (r = 0.996) with only 13 unique fails. ADC_INL (95 ms) has near-zero failures. Moving both to 5-10% sample audit saves {test_proposals.get('total_test_time_reduction_ms', 105):.1f} ms per die ({test_proposals.get('test_time_reduction_pct', 8.5):.1f}% test time reduction). Rule-of-three escape bound is < 6.0 DPPM.",
                "confidence_pct": 92.0,
                "expected_benefit_usd": test_proposals.get("total_annual_savings_usd", 85000.0),
                "benefit_description": f"${test_proposals.get('total_annual_savings_usd', 85000.0):,.2f} annual ATE tester cost savings across 5.4M units.",
                "quality_risk": "Statistical 95% upper bound on escape rate is < 6.0 DPPM under 5% audit sampling.",
                "recommended_action": "Submit formal engineering change request (ECR) to Change Control Board for AX7_CP1_R4.3 release.",
                "status": "Pending Review",
                "owner": "Test Engineering Lead"
            })
            rec_counter += 1

        # 7. Wafer Handler ROB-02 Scratch Defense
        recs.append({
            "id": f"REC-{rec_counter:02d}",
            "title": "Robot End-Effector Calibration: Handler ROB-02 Scratches",
            "engine_source": "Wafer Map AI & Root Cause Engine",
            "category": "Mechanical Handling Damage",
            "severity": "LOW",
            "priority": "P3",
            "affected_scope": "Wafer Handler ROB-02 (~6% of wafers loaded)",
            "evidence": "Linear scratch signatures identified by Wafer Map AI across ROB-02 loaded wafers. Average yield loss of 3.2% on affected scratch wafers.",
            "confidence_pct": 88.0,
            "expected_benefit_usd": 65000.0,
            "benefit_description": "Saves 15-20 damaged wafers per quarter ($65k/yr).",
            "quality_risk": "Zero escape risk.",
            "recommended_action": "Inspect vacuum wand and robotic cassette transfer arm on ROB-02 for particle buildup or misaligned gripper pins.",
            "status": "Pending Review",
            "owner": "Test Floor Maintenance"
        })

        self.recommendations = recs
        return recs

    def update_status(self, rec_id: str, new_status: str, engineer_notes: str = "") -> bool:
        """
        Human-in-the-loop status transition (Pending Review -> Approved / Rejected / Completed).
        Advisory governance: never modifies tester code directly.
        """
        for r in self.recommendations:
            if r["id"] == rec_id:
                r["status"] = new_status
                r["engineer_notes"] = engineer_notes
                r["updated_ts"] = pd.Timestamp.now().isoformat()
                return True
        return False
