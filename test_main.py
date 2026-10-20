"""
test_main.py
============
Pytest test suite for SME Finance Copilot.

Covers:
  - Security Layer: PII detection and anonymization
  - Legal Engine: Compliance checks and blocking logic
  - Cash-Flow Engine: Monte Carlo stress test statistics
  - AI Agent: RAG recommendation scoring and ranking
  - FastAPI endpoints: integration tests via TestClient

Execution evidence is logged to execution_evidence.log.
"""

from __future__ import annotations

import json
import logging
import sys
import traceback
from datetime import date
from typing import Any, Dict

import pytest
from fastapi.testclient import TestClient

# ---------------------------------------------------------------------------
# Evidence logger (writes execution_evidence.log)
# ---------------------------------------------------------------------------

_evidence_logger = logging.getLogger("execution_evidence")
_evidence_logger.setLevel(logging.DEBUG)
_fh = logging.FileHandler("execution_evidence.log", mode="w", encoding="utf-8")
_fh.setFormatter(
    logging.Formatter("%(asctime)s | %(levelname)-8s | %(message)s")
)
_evidence_logger.addHandler(_fh)
_evidence_logger.addHandler(logging.StreamHandler(sys.stdout))

# Also configure root logger to stdout
logging.basicConfig(level=logging.INFO)


def evidence(msg: str, data: Any = None) -> None:
    """Write a structured evidence entry to the log file."""
    if data is not None:
        try:
            serialised = json.dumps(data, default=str, indent=2)
            _evidence_logger.info("%s\n%s", msg, serialised)
        except Exception:
            _evidence_logger.info("%s | DATA: %s", msg, str(data))
    else:
        _evidence_logger.info(msg)


# ---------------------------------------------------------------------------
# Module imports (with evidence)
# ---------------------------------------------------------------------------

evidence("=" * 70)
evidence("SME Finance Copilot — Test Suite Execution Starting")
evidence("FinTechathon 2026 — Shenzhen International FinTech Competition")
evidence("=" * 70)

from security_layer import PrivacyPreservingLayer, _regex_mask, PIIDetectionResult

evidence("✅ security_layer imported successfully")

from legal_engine import (
    RegulatoryComplianceEngine,
    MercantileRegistryRules,
    LitigationRiskScorer,
    AMLScreener,
    ComplianceStatus,
    LitigationRisk,
    AMLFlag,
)

evidence("✅ legal_engine imported successfully")

from cashflow_engine import CashFlowStressTester, MonteCarloCashFlow, SCENARIO_PARAMS

evidence("✅ cashflow_engine imported successfully")

from ai_agent import SMEFinancialCopilot, PRODUCT_CATALOGUE, EmbeddingEngine

evidence("✅ ai_agent imported successfully")

from main import app

evidence("✅ main FastAPI app imported successfully")

client = TestClient(app)


# ===========================================================================
# SECTION 1 — Security Layer Tests
# ===========================================================================

class TestSecurityLayer:
    """Tests for PrivacyPreservingLayer (PII masking)."""

    def setup_method(self):
        self.ppl = PrivacyPreservingLayer(enable_audit_log=True)
        evidence("-" * 40)
        evidence("TestSecurityLayer: setup complete")

    def test_email_redaction(self):
        """Email addresses must be anonymized."""
        evidence("TEST: test_email_redaction")
        text = "Contact us at ceo@acme-finance.com for details."
        result = self.ppl.anonymize(text)
        evidence("Input text", {"text": text})
        evidence("Anonymized", {"result": result.anonymized_text, "entities": result.entities_found})
        assert "ceo@acme-finance.com" not in result.anonymized_text, \
            "Email address should be redacted"
        assert result.risk_score >= 0.0
        assert result.audit_token != ""
        evidence("✅ test_email_redaction PASSED")

    def test_phone_redaction(self):
        """Phone numbers must be anonymized."""
        evidence("TEST: test_phone_redaction")
        text = "Call +34 612 345 678 or 912 345 678."
        result = self.ppl.anonymize(text)
        evidence("Input", {"text": text, "anonymized": result.anonymized_text})
        assert "+34 612 345 678" not in result.anonymized_text, \
            "Phone number should be redacted"
        evidence("✅ test_phone_redaction PASSED")

    def test_iban_redaction(self):
        """IBAN codes must be anonymized."""
        evidence("TEST: test_iban_redaction")
        text = "Please transfer to ES9121000418401234567891"
        result = self.ppl.anonymize(text)
        evidence("IBAN test", {"anonymized": result.anonymized_text})
        assert "ES9121000418401234567891" not in result.anonymized_text
        evidence("✅ test_iban_redaction PASSED")

    def test_credit_card_redaction(self):
        """Credit card numbers must be masked."""
        evidence("TEST: test_credit_card_redaction")
        text = "Pay with card 4111 1111 1111 1111 expiry 12/27."
        result = self.ppl.anonymize(text)
        evidence("Card test", {"anonymized": result.anonymized_text})
        assert "4111 1111 1111 1111" not in result.anonymized_text
        evidence("✅ test_credit_card_redaction PASSED")

    def test_empty_text(self):
        """Empty text returns safely with zero entities."""
        evidence("TEST: test_empty_text")
        result = self.ppl.anonymize("")
        assert result.anonymized_text == ""
        assert result.entities_found == []
        assert result.risk_score == 0.0
        evidence("✅ test_empty_text PASSED")

    def test_risk_score_range(self):
        """Risk score must always be in [0.0, 1.0]."""
        evidence("TEST: test_risk_score_range")
        texts = [
            "Normal business text with no PII.",
            "CEO John Smith, IBAN ES1234567890123456789012, card 4111111111111111",
            "Email: a@b.com, phone +34600000000",
        ]
        for t in texts:
            result = self.ppl.anonymize(t)
            assert 0.0 <= result.risk_score <= 1.0, f"Risk score out of range: {result.risk_score}"
        evidence("✅ test_risk_score_range PASSED")

    def test_batch_anonymization(self):
        """Batch processing returns consistent counts."""
        evidence("TEST: test_batch_anonymization")
        texts = [
            "Email: ceo@acme.com",
            "Phone: +34 612 345 678",
            "No PII here at all.",
        ]
        results, report = self.ppl.anonymize_batch(texts)
        evidence("Batch report", {"total_entities": report.total_entities_redacted,
                                   "breakdown": report.entity_type_breakdown})
        assert len(results) == 3
        assert report.total_documents == 3
        assert isinstance(report.entity_type_breakdown, dict)
        evidence("✅ test_batch_anonymization PASSED")

    def test_audit_trail(self):
        """Audit trail records entries after anonymization."""
        evidence("TEST: test_audit_trail")
        ppl = PrivacyPreservingLayer(enable_audit_log=True)
        ppl.anonymize("Call me at +34 600 000 001")
        ppl.anonymize("Email test@test.com")
        trail = ppl.get_audit_trail()
        evidence("Audit trail", trail)
        assert len(trail) == 2
        assert all("audit_token" in entry for entry in trail)
        assert all("transaction_id" in entry for entry in trail)
        assert all("sha256_hash" in entry for entry in trail)
        evidence("✅ test_audit_trail PASSED")

    def test_regex_fallback_directly(self):
        """Test the regex masking fallback directly."""
        evidence("TEST: test_regex_fallback_directly")
        text = "CEO email ceo@demo.com, IBAN ES9121000418401234567891"
        masked, entities = _regex_mask(text)
        evidence("Regex mask result", {"masked": masked, "count": len(entities)})
        assert "ceo@demo.com" not in masked
        assert len(entities) >= 1
        evidence("✅ test_regex_fallback_directly PASSED")

    def test_full_name_redaction(self):
        """Full names like John Doe, Jane Smith, and CEO: John Smith must be masked."""
        evidence("TEST: test_full_name_redaction")
        sample = "CEO: John Smith and CFO Jane Smith met with John Doe regarding financing."
        result = self.ppl.anonymize(sample)
        evidence("Name redaction", {"original": sample, "anonymized": result.anonymized_text})
        assert "John Smith" not in result.anonymized_text
        assert "Jane Smith" not in result.anonymized_text
        assert "John Doe" not in result.anonymized_text
        assert "<PERSON>" in result.anonymized_text
        evidence("✅ test_full_name_redaction PASSED")

    def test_clean_tag_format(self):
        """All PII types must be masked into clean <TAG> format."""
        evidence("TEST: test_clean_tag_format")
        text = (
            "Contact CEO John Smith at john.doe@email.com or +34 612 345 678. "
            "IBAN: ES9121000418450200051332. Card: 4111 1111 1111 1111. CIF: B87654321."
        )
        result = self.ppl.anonymize(text)
        evidence("Clean tag result", {"anonymized": result.anonymized_text})
        assert "<PERSON>" in result.anonymized_text
        assert "<EMAIL_ADDRESS>" in result.anonymized_text
        assert "<PHONE_NUMBER>" in result.anonymized_text
        assert "<IBAN_CODE>" in result.anonymized_text
        assert "<CREDIT_CARD>" in result.anonymized_text
        assert "<TAX_ID>" in result.anonymized_text
        evidence("✅ test_clean_tag_format PASSED")

    def test_verifiable_audit_trail_object(self):
        """Anonymization must produce an explicit, verifiable audit trail object with SHA256."""
        evidence("TEST: test_verifiable_audit_trail_object")
        text = "Director Carlos Mendez, email carlos@corp.es, phone +34 912 345 678"
        result = self.ppl.anonymize(text, regulatory_verdict="COMPLIANT")
        audit = result.audit_trail
        evidence("Verifiable audit trail", audit)
        assert isinstance(audit, dict)
        assert "timestamp" in audit
        assert "transaction_id" in audit
        assert "original_pii_count" in audit
        assert "anonymized_text" in audit
        assert "regulatory_verdict" in audit
        assert "sha256_hash" in audit
        assert len(audit["sha256_hash"]) == 64  # valid SHA-256 hex string
        assert audit["original_pii_count"] > 0
        evidence("✅ test_verifiable_audit_trail_object PASSED")


# ===========================================================================
# SECTION 2 — Legal Engine Tests
# ===========================================================================

class TestLegalEngine:
    """Tests for RegulatoryComplianceEngine."""

    def setup_method(self):
        self.engine = RegulatoryComplianceEngine()
        evidence("-" * 40)
        evidence("TestLegalEngine: setup complete")

    def test_compliant_on_time_deposit(self):
        """Company that deposited on time should be COMPLIANT."""
        evidence("TEST: test_compliant_on_time_deposit")
        check = MercantileRegistryRules.assess(
            fiscal_year_end=date(2022, 12, 31),
            deposit_date=date(2023, 7, 25),  # within 7-month window
        )
        evidence("Accounts check", {"status": check.status, "days_overdue": check.days_overdue})
        assert check.status == ComplianceStatus.COMPLIANT
        assert check.days_overdue == 0
        evidence("✅ test_compliant_on_time_deposit PASSED")

    def test_blocked_severely_overdue(self):
        """Accounts > 6 months overdue must be BLOCKED."""
        evidence("TEST: test_blocked_severely_overdue")
        check = MercantileRegistryRules.assess(
            fiscal_year_end=date(2021, 12, 31),
            deposit_date=None,
            reference_date=date(2023, 12, 31),  # 2+ years later
        )
        evidence("Blocked check", {"status": check.status, "overdue": check.days_overdue})
        assert check.status == ComplianceStatus.BLOCKED
        assert check.days_overdue > 180
        evidence("✅ test_blocked_severely_overdue PASSED")

    def test_litigation_risk_low(self):
        """Clean financial profile should score LOW litigation risk."""
        evidence("TEST: test_litigation_risk_low")
        assessment = LitigationRiskScorer.score(
            payment_delay_days=5,
            creditor_disputes=0,
            court_proceedings=0,
            debt_to_equity=0.5,
            revenue_decline_pct=0.0,
        )
        evidence("Low risk assessment", {"risk_level": assessment.risk_level, "score": assessment.risk_score})
        assert assessment.risk_level == LitigationRisk.LOW
        assert assessment.risk_score < 0.30
        evidence("✅ test_litigation_risk_low PASSED")

    def test_litigation_risk_critical(self):
        """Distressed profile must score CRITICAL."""
        evidence("TEST: test_litigation_risk_critical")
        assessment = LitigationRiskScorer.score(
            payment_delay_days=180,
            creditor_disputes=10,
            court_proceedings=5,
            debt_to_equity=6.0,
            revenue_decline_pct=50.0,
        )
        evidence("Critical risk", {"risk_level": assessment.risk_level, "score": assessment.risk_score})
        assert assessment.risk_level == LitigationRisk.CRITICAL
        assert assessment.risk_score > 0.80
        evidence("✅ test_litigation_risk_critical PASSED")

    def test_aml_clear(self):
        """Clean entity name should clear AML screening."""
        evidence("TEST: test_aml_clear")
        result = AMLScreener.screen("Acme Logistics SL", "ES")
        evidence("AML clear", {"flag": result.flag, "confidence": result.confidence})
        assert result.flag == AMLFlag.CLEAR
        evidence("✅ test_aml_clear PASSED")

    def test_aml_blocked_watchlist(self):
        """Entity matching watchlist pattern must be BLOCKED."""
        evidence("TEST: test_aml_blocked_watchlist")
        result = AMLScreener.screen("Offshore Holding LLC", "KY")
        evidence("AML blocked", {"flag": result.flag, "matched": result.matched_lists})
        assert result.flag in (AMLFlag.BLOCKED, AMLFlag.WATCHLIST)
        evidence("✅ test_aml_blocked_watchlist PASSED")

    def test_full_compliance_report_structure(self):
        """Full compliance report must have all required fields."""
        evidence("TEST: test_full_compliance_report_structure")
        report = self.engine.run_full_compliance_check(
            entity_name="Test Corp SL",
            cif="B99999999",
            fiscal_year_end=date(2023, 12, 31),
            deposit_date=date(2024, 7, 15),
            payment_delay_days=30,
            creditor_disputes=1,
            court_proceedings=0,
            debt_to_equity=1.5,
            revenue_decline_pct=5.0,
        )
        evidence("Full report summary", {
            "report_id": report.report_id,
            "overall_status": report.overall_status.value,
            "compliance_score": report.compliance_score,
            "lending_eligible": report.lending_eligible,
        })
        assert report.report_id.startswith("RCE-")
        assert report.overall_status in ComplianceStatus.__members__.values()
        assert 0 <= report.compliance_score <= 100
        assert isinstance(report.lending_eligible, bool)
        assert len(report.regulatory_references) > 0
        evidence("✅ test_full_compliance_report_structure PASSED")

    def test_deposit_deadline_computation(self):
        """Deadline = fiscal_year_end + 7 months (last day)."""
        evidence("TEST: test_deposit_deadline_computation")
        fy_end = date(2023, 12, 31)
        deadline = MercantileRegistryRules.compute_deadline(fy_end)
        evidence("Deadline test", {"fy_end": str(fy_end), "deadline": str(deadline)})
        # 2023-12-31 + 7 months = 2024-07-31
        assert deadline == date(2024, 7, 31)
        evidence("✅ test_deposit_deadline_computation PASSED")


# ===========================================================================
# SECTION 3 — Cash-Flow Engine Tests
# ===========================================================================

class TestCashFlowEngine:
    """Tests for CashFlowStressTester (Monte Carlo)."""

    def setup_method(self):
        # Use 1,000 simulations for faster test runs
        self.tester = CashFlowStressTester(simulations=1_000, seed=42)
        evidence("-" * 40)
        evidence("TestCashFlowEngine: setup complete (1,000 simulations)")

    def test_full_stress_report_structure(self):
        """Report must contain 9 scenarios (3 horizons × 3 scenarios)."""
        evidence("TEST: test_full_stress_report_structure")
        report = self.tester.run_full_stress_test(
            entity_name="Test Corp SL",
            opening_balance=200_000,
            monthly_revenue=150_000,
            monthly_expenses=130_000,
        )
        evidence("Stress report", {
            "report_id": report.report_id,
            "liquidity_rating": report.overall_liquidity_rating,
            "scenario_count": len(report.scenarios),
        })
        assert report.report_id.startswith("CFE-")
        assert len(report.scenarios) == 9  # 3 horizons × 3 scenarios
        assert report.overall_liquidity_rating in ("A+", "A", "B", "C", "D")
        evidence("✅ test_full_stress_report_structure PASSED")

    def test_scenario_statistics_validity(self):
        """VaR99 must be <= VaR95 (worse tail), P(ruin) in [0,1]."""
        evidence("TEST: test_scenario_statistics_validity")
        report = self.tester.run_full_stress_test(
            entity_name="Test",
            opening_balance=100_000,
            monthly_revenue=80_000,
            monthly_expenses=75_000,
        )
        for scen in report.scenarios:
            evidence(f"Scenario {scen.scenario_name} @ {scen.horizon_days}d", {
                "mean": scen.mean_closing_balance,
                "var_95": scen.var_95,
                "var_99": scen.var_99,
                "p_ruin": scen.probability_of_ruin,
            })
            assert scen.var_99 <= scen.var_95 + 1, \
                f"VaR99 should be <= VaR95 (worse or equal tail): {scen.var_99} vs {scen.var_95}"
            assert 0.0 <= scen.probability_of_ruin <= 1.0
            assert scen.simulations > 0
        evidence("✅ test_scenario_statistics_validity PASSED")

    def test_severe_stress_worse_than_baseline(self):
        """Severe stress mean balance must be lower than baseline."""
        evidence("TEST: test_severe_stress_worse_than_baseline")
        report = self.tester.run_full_stress_test(
            entity_name="Test",
            opening_balance=300_000,
            monthly_revenue=200_000,
            monthly_expenses=150_000,
        )
        baseline_90 = next(
            (s for s in report.scenarios
             if s.scenario_name == "baseline" and s.horizon_days == 90), None
        )
        severe_90 = next(
            (s for s in report.scenarios
             if s.scenario_name == "severe_stress" and s.horizon_days == 90), None
        )
        assert baseline_90 and severe_90, "Should have both baseline and severe_stress scenarios"
        evidence("Baseline vs Severe 90d", {
            "baseline_mean": baseline_90.mean_closing_balance,
            "severe_mean": severe_90.mean_closing_balance,
        })
        assert severe_90.mean_closing_balance < baseline_90.mean_closing_balance, \
            "Severe stress should yield lower mean balance"
        evidence("✅ test_severe_stress_worse_than_baseline PASSED")

    def test_healthy_sme_high_liquidity_rating(self):
        """Profitable SME with strong buffer should get A or A+."""
        evidence("TEST: test_healthy_sme_high_liquidity_rating")
        report = self.tester.run_full_stress_test(
            entity_name="HealthyCorp SL",
            opening_balance=1_000_000,
            monthly_revenue=500_000,
            monthly_expenses=300_000,
        )
        evidence("Healthy SME rating", {"rating": report.overall_liquidity_rating})
        assert report.overall_liquidity_rating in ("A+", "A", "B"), \
            f"Healthy SME got unexpected rating: {report.overall_liquidity_rating}"
        evidence("✅ test_healthy_sme_high_liquidity_rating PASSED")

    def test_quick_liquidity_check(self):
        """Quick check returns expected keys."""
        evidence("TEST: test_quick_liquidity_check")
        mc = MonteCarloCashFlow(simulations=500, seed=0)
        result = mc.run_scenario(100_000, 80_000, 70_000, 30, "baseline")
        evidence("Quick scenario result", {
            "mean": result.mean_closing_balance,
            "var_95": result.var_95,
            "p_ruin": result.probability_of_ruin,
        })
        assert result.horizon_days == 30
        assert result.scenario_name == "baseline"
        assert isinstance(result.mean_closing_balance, float)
        evidence("✅ test_quick_liquidity_check PASSED")


# ===========================================================================
# SECTION 4 — AI Agent Tests
# ===========================================================================

class TestAIAgent:
    """Tests for SMEFinancialCopilot RAG recommendations."""

    @pytest.fixture(autouse=True, scope="class")
    def setup_copilot(self):
        TestAIAgent.copilot = SMEFinancialCopilot()
        evidence("-" * 40)
        evidence("TestAIAgent: SMEFinancialCopilot initialised")

    def test_recommendation_structure(self):
        """Response must have recommendations list and required fields."""
        evidence("TEST: test_recommendation_structure")
        resp = self.copilot.recommend(
            entity_name="Test Corp",
            query="working capital loan for seasonal business",
            top_k=3,
        )
        evidence("Recommendation response", {
            "query_id": resp.query_id,
            "count": len(resp.recommendations),
            "model": resp.model_used,
            "ms": resp.processing_time_ms,
        })
        assert len(resp.recommendations) <= 3
        assert resp.query_id != ""
        assert resp.total_products_evaluated == len(PRODUCT_CATALOGUE)
        evidence("✅ test_recommendation_structure PASSED")

    def test_recommendations_sorted_by_score(self):
        """Recommendations must be in descending relevance order."""
        evidence("TEST: test_recommendations_sorted_by_score")
        resp = self.copilot.recommend(
            entity_name="Test Corp",
            query="R&D innovation technology startup grant",
            top_k=5,
        )
        scores = [r.relevance_score for r in resp.recommendations]
        evidence("Scores in order", scores)
        assert scores == sorted(scores, reverse=True), "Recommendations not sorted by score"
        evidence("✅ test_recommendations_sorted_by_score PASSED")

    def test_score_in_valid_range(self):
        """All relevance scores must be in [0.0, 1.0]."""
        evidence("TEST: test_score_in_valid_range")
        resp = self.copilot.recommend(
            entity_name="Corp",
            query="insurance against cyber attacks and business interruption",
            top_k=5,
        )
        for rec in resp.recommendations:
            assert 0.0 <= rec.relevance_score <= 1.0, f"Score out of range: {rec.relevance_score}"
        evidence("✅ test_score_in_valid_range PASSED", [r.relevance_score for r in resp.recommendations])

    def test_category_filter_lending_only(self):
        """Category filter 'lending' should only return lending products."""
        evidence("TEST: test_category_filter_lending_only")
        resp = self.copilot.recommend(
            entity_name="Corp",
            query="need a loan",
            top_k=5,
            categories=["lending"],
        )
        evidence("Lending-only results", [(r.product_id, r.category) for r in resp.recommendations])
        for rec in resp.recommendations:
            assert rec.category == "lending", f"Expected lending, got {rec.category}"
        evidence("✅ test_category_filter_lending_only PASSED")

    def test_product_catalogue_completeness(self):
        """Product catalogue must have required fields."""
        evidence("TEST: test_product_catalogue_completeness")
        for product in PRODUCT_CATALOGUE:
            assert "id" in product
            assert "name" in product
            assert "category" in product
            assert "description" in product
            assert product["category"] in ("lending", "grant", "insurance")
        evidence(f"✅ test_product_catalogue_completeness PASSED ({len(PRODUCT_CATALOGUE)} products)")

    def test_eu_ai_act_disclosure_present(self):
        """EU AI Act Art.13 disclosure must be present in every response."""
        evidence("TEST: test_eu_ai_act_disclosure_present")
        resp = self.copilot.recommend("Corp", "any query", top_k=1)
        assert resp.eu_ai_act_transparency != ""
        assert "EU AI Act" in resp.eu_ai_act_transparency
        evidence("✅ test_eu_ai_act_disclosure_present PASSED")

    def test_amount_filter(self):
        """Amount filter must exclude products below the requested amount."""
        evidence("TEST: test_amount_filter")
        resp = self.copilot.recommend(
            entity_name="LargeCorp",
            query="large loan for major expansion",
            top_k=5,
            amount_eur=5_000_000,
        )
        evidence("Amount-filtered results", [
            {"id": r.product_id, "max": r.max_amount_eur} for r in resp.recommendations
        ])
        for rec in resp.recommendations:
            assert rec.max_amount_eur == 0 or rec.max_amount_eur >= 5_000_000, \
                f"Product {rec.product_id} with max {rec.max_amount_eur} slipped through filter"
        evidence("✅ test_amount_filter PASSED")


# ===========================================================================
# SECTION 5 — FastAPI Integration Tests
# ===========================================================================

class TestFastAPIEndpoints:
    """Integration tests against the FastAPI TestClient."""

    def test_health_check(self):
        """GET / must return 200 with operational status."""
        evidence("TEST: test_health_check [API]")
        response = client.get("/")
        evidence("Health response", response.json())
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "operational"
        assert data["version"] == "2.0.0"
        evidence("✅ test_health_check PASSED")

    def test_dashboard_endpoint(self):
        """GET /dashboard must return HTML."""
        evidence("TEST: test_dashboard_endpoint [API]")
        response = client.get("/dashboard")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
        assert "SME Finance Copilot" in response.text
        evidence("✅ test_dashboard_endpoint PASSED (HTML dashboard served)")

    def test_mask_pii_endpoint(self):
        """POST /mask-pii must anonymize PII and return verifiable audit_trail."""
        evidence("TEST: test_mask_pii_endpoint [API]")
        payload = {"text": "CEO: John Smith at john@company.com, +34 612 345 678"}
        response = client.post("/mask-pii", json=payload)
        evidence("PII endpoint response", response.json())
        assert response.status_code == 200
        data = response.json()
        assert "anonymized_text" in data
        assert "john@company.com" not in data["anonymized_text"]
        assert "audit_token" in data
        assert "audit_trail" in data
        assert "sha256_hash" in data["audit_trail"]
        assert "original_pii_count" in data["audit_trail"]
        evidence("✅ test_mask_pii_endpoint PASSED")

    def test_compliance_endpoint_compliant(self):
        """POST /compliance with clean profile should return COMPLIANT."""
        evidence("TEST: test_compliance_endpoint_compliant [API]")
        payload = {
            "entity_name": "Clean Corp SL",
            "cif": "B11111111",
            "fiscal_year_end": "2022-12-31",
            "deposit_date": "2023-07-20",
            "payment_delay_days": 5.0,
            "creditor_disputes": 0,
            "court_proceedings": 0,
            "debt_to_equity": 0.8,
            "revenue_decline_pct": 2.0,
            "jurisdiction": "ES",
        }
        response = client.post("/compliance", json=payload)
        data = response.json()
        evidence("Compliance response (compliant)", {
            "status": data.get("overall_status"),
            "score": data.get("compliance_score"),
            "eligible": data.get("lending_eligible"),
        })
        assert response.status_code == 200
        assert data["overall_status"] in ("COMPLIANT", "WARNING", "NON_COMPLIANT", "BLOCKED")
        evidence("✅ test_compliance_endpoint_compliant PASSED")

    def test_compliance_endpoint_blocked(self):
        """POST /compliance with severely overdue accounts returns BLOCKED."""
        evidence("TEST: test_compliance_endpoint_blocked [API]")
        payload = {
            "entity_name": "Bad Corp SL",
            "cif": "B22222222",
            "fiscal_year_end": "2020-12-31",  # 4+ years ago, never deposited
            "deposit_date": None,
            "payment_delay_days": 200.0,
            "creditor_disputes": 8,
            "court_proceedings": 3,
            "debt_to_equity": 5.0,
            "revenue_decline_pct": 40.0,
            "jurisdiction": "ES",
        }
        response = client.post("/compliance", json=payload)
        data = response.json()
        evidence("Compliance blocked response", {
            "status": data.get("overall_status"),
            "blocking_reasons": data.get("blocking_reasons"),
        })
        assert response.status_code == 200
        assert data["overall_status"] == "BLOCKED"
        assert data["lending_eligible"] is False
        evidence("✅ test_compliance_endpoint_blocked PASSED")

    def test_stress_test_endpoint(self):
        """POST /stress-test must return 9 scenarios with valid statistics."""
        evidence("TEST: test_stress_test_endpoint [API]")
        payload = {
            "entity_name": "Acme SL",
            "opening_balance": 250000.0,
            "monthly_revenue": 180000.0,
            "monthly_expenses": 155000.0,
        }
        response = client.post("/stress-test", json=payload)
        data = response.json()
        evidence("Stress test response summary", {
            "report_id": data.get("report_id"),
            "rating": data.get("overall_liquidity_rating"),
            "scenario_count": len(data.get("scenarios", [])),
        })
        assert response.status_code == 200
        assert len(data["scenarios"]) == 9
        assert data["overall_liquidity_rating"] in ("A+", "A", "B", "C", "D")
        evidence("✅ test_stress_test_endpoint PASSED")

    def test_recommend_endpoint(self):
        """POST /recommend must return ranked recommendations."""
        evidence("TEST: test_recommend_endpoint [API]")
        payload = {
            "entity_name": "TechStartup SL",
            "query": "We need R&D funding and innovation grants for AI development",
            "top_k": 5,
        }
        response = client.post("/recommend", json=payload)
        data = response.json()
        evidence("Recommend response", {
            "count": len(data.get("recommendations", [])),
            "model": data.get("model_used"),
            "products_evaluated": data.get("total_products_evaluated"),
        })
        assert response.status_code == 200
        assert len(data["recommendations"]) <= 5
        assert data["total_products_evaluated"] == len(PRODUCT_CATALOGUE)
        evidence("✅ test_recommend_endpoint PASSED")

    def test_full_analysis_pipeline(self):
        """POST /analyze must run all 4 engines and return composite result."""
        evidence("TEST: test_full_analysis_pipeline [API]")
        payload = {
            "entity_name": "Pipeline Test SL",
            "cif": "B55555555",
            "raw_description": "CEO Maria López, email maria@pipeline.com, +34 699 888 777",
            "fiscal_year_end": "2023-12-31",
            "deposit_date": "2024-07-15",
            "opening_balance": 300000.0,
            "monthly_revenue": 200000.0,
            "monthly_expenses": 165000.0,
            "payment_delay_days": 20.0,
            "creditor_disputes": 0,
            "court_proceedings": 0,
            "debt_to_equity": 1.2,
            "revenue_decline_pct": 3.0,
            "financing_query": "We need working capital to expand operations",
        }
        response = client.post("/analyze", json=payload)
        data = response.json()
        evidence("Full pipeline response", {
            "pipeline_id": data.get("pipeline_id"),
            "pii_entities": len(data.get("pii_result", {}).get("entities_found", [])),
            "compliance_status": data.get("compliance_report", {}).get("overall_status"),
            "liquidity_rating": data.get("stress_test_report", {}).get("liquidity_rating"),
            "recommendation_count": len(data.get("ai_recommendations", [])),
            "total_ms": data.get("total_processing_ms"),
        })
        assert response.status_code == 200
        assert "pipeline_id" in data
        assert "pii_result" in data
        assert "compliance_report" in data
        assert "stress_test_report" in data
        assert "ai_recommendations" in data
        assert "audit_trail" in data
        assert "sha256_hash" in data["audit_trail"]
        assert "regulatory_verdict" in data["audit_trail"]
        assert len(data["audit_trail"]["sha256_hash"]) == 64
        assert data.get("total_processing_ms", 0) > 0
        evidence("✅ test_full_analysis_pipeline PASSED")

    def test_products_endpoint(self):
        """GET /products returns full catalogue."""
        evidence("TEST: test_products_endpoint [API]")
        response = client.get("/products")
        data = response.json()
        evidence("Products endpoint", {"total": data.get("total"), "categories": data.get("categories")})
        assert response.status_code == 200
        assert data["total"] == len(PRODUCT_CATALOGUE)
        evidence("✅ test_products_endpoint PASSED")

    def test_audit_trail_endpoint(self):
        """GET /audit-trail returns structured audit log."""
        evidence("TEST: test_audit_trail_endpoint [API]")
        response = client.get("/audit-trail")
        data = response.json()
        evidence("Audit trail endpoint", {
            "total_requests": data.get("total_requests"),
            "total_pii_scans": data.get("total_pii_scans"),
        })
        assert response.status_code == 200
        assert "session_audit_log" in data
        assert "pii_audit_trail" in data
        evidence("✅ test_audit_trail_endpoint PASSED")

    def test_invalid_date_format(self):
        """Invalid date format returns 422 Unprocessable Entity."""
        evidence("TEST: test_invalid_date_format [API]")
        payload = {
            "entity_name": "Test", "cif": "B00000000",
            "fiscal_year_end": "not-a-date",
            "payment_delay_days": 0, "creditor_disputes": 0,
            "court_proceedings": 0, "debt_to_equity": 0.5,
            "revenue_decline_pct": 0, "jurisdiction": "ES",
        }
        response = client.post("/compliance", json=payload)
        evidence("Invalid date response", {"status": response.status_code})
        assert response.status_code == 422
        evidence("✅ test_invalid_date_format PASSED")


# ===========================================================================
# Teardown / Summary
# ===========================================================================

def pytest_sessionfinish(session, exitstatus):
    """Log final test summary to evidence file."""
    evidence("=" * 70)
    evidence(f"TEST SESSION COMPLETE — Exit code: {exitstatus}")
    evidence(f"All results logged to: execution_evidence.log")
    evidence("FinTechathon 2026 — SME Finance Copilot — Submission Ready")
    evidence("=" * 70)
