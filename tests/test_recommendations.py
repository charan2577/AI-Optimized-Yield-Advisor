"""
Unit tests for Recommendation Engine, prioritization, and status transitions.
"""

import pytest
import config
from src.recommendation_engine import RecommendationEngine


def test_recommendation_generation_and_scoring():
    """Verify recommendation generation and status update workflow."""
    rec_engine = RecommendationEngine()
    recs = rec_engine.generate_all_recommendations(
        excursion_events=[{"target_id": "ETCH-07 / Chamber C"}],
        root_causes=[{"factor": "PC-4412"}],
        latent_results={"dppm_reduction_pct": 40.1, "yield_cost_pts": 0.49},
        test_proposals={"proposals": [{"test_name": "RING_OSC", "annual_savings_usd": 16000}]},
        retest_results={"by_site": True}
    )

    assert len(recs) >= 5
    # Verify required fields
    for r in recs:
        assert "id" in r
        assert "title" in r
        assert "severity" in r
        assert "priority" in r
        assert "expected_benefit_usd" in r
        assert "status" in r
        assert r["status"] == "Pending Review"

    # Test status update workflow
    first_id = recs[0]["id"]
    updated = rec_engine.update_status(first_id, "Approved", engineer_notes="CCB approved ECR-2026-09")
    assert updated is True
    assert rec_engine.recommendations[0]["status"] == "Approved"
    assert rec_engine.recommendations[0]["engineer_notes"] == "CCB approved ECR-2026-09"
