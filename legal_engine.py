"""
legal_engine.py
===============
Mercantile Registry Compliance Engine — RegulatoryComplianceEngine.

Checks:
  1. Annual-accounts deposit deadlines (Spanish Mercantile Registry rules).
  2. Litigation risk scoring based on payment delays, creditor disputes, etc.
  3. CNMV / Bank of Spain prudential thresholds.
  4. AML/KYB red-flag screening.

Compliance references:
  - Real Decreto Legislativo 1/2010 (Ley de Sociedades de Capital, Art. 279)
  - Ley 22/2003 de Concurso de Acreedores (insolvency triggers)
  - EU AML Directive 6AMLD
  - Bank of Spain Circular 4/2017
"""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class ComplianceStatus(str, Enum):
    COMPLIANT       = "COMPLIANT"
    WARNING         = "WARNING"
    NON_COMPLIANT   = "NON_COMPLIANT"
    BLOCKED         = "BLOCKED"          # regulatory block — no lending


class LitigationRisk(str, Enum):
    LOW      = "LOW"
    MEDIUM   = "MEDIUM"
    HIGH     = "HIGH"
    CRITICAL = "CRITICAL"


class AMLFlag(str, Enum):
    CLEAR    = "CLEAR"
    WATCHLIST = "WATCHLIST"
    BLOCKED  = "BLOCKED"


# ---------------------------------------------------------------------------
# Data-classes
# ---------------------------------------------------------------------------

@dataclass
class AnnualAccountsCheck:
    fiscal_year_end: date
    deposit_deadline: date
    deposit_date: Optional[date]
    days_overdue: int
    status: ComplianceStatus
    penalty_risk_eur: float
    notes: str


@dataclass
class LitigationRiskAssessment:
    risk_level: LitigationRisk
    risk_score: float          # 0.0 – 1.0
    flags: List[str]
    contributing_factors: Dict[str, float]
    recommendation: str


@dataclass
class AMLScreeningResult:
    entity_name: str
    flag: AMLFlag
    matched_lists: List[str]
    confidence: float
    details: str


@dataclass
class ComplianceReport:
    """Full regulatory compliance report for a single SME."""
    report_id: str
    entity_name: str
    cif: str
    generated_at: str
    overall_status: ComplianceStatus
    annual_accounts: AnnualAccountsCheck
    litigation: LitigationRiskAssessment
    aml_screening: AMLScreeningResult
    lending_eligible: bool
    blocking_reasons: List[str]
    compliance_score: float    # 0–100 composite
    next_review_date: str
    regulatory_references: List[str]


# ---------------------------------------------------------------------------
# Spanish Mercantile Registry rules
# ---------------------------------------------------------------------------

class MercantileRegistryRules:
    """
    Encodes Spanish corporate law deposit deadlines.

    Key rule (LSC Art. 279):
      - Accounts must be approved by GM within 6 months of fiscal year-end.
      - Deposit in Mercantile Registry must occur within 1 month of GM approval.
      - Effective deposit deadline = fiscal_year_end + 7 months (standard path).
    """

    STANDARD_MONTHS_TO_DEPOSIT = 7        # months after fiscal year-end
    MAX_OVERDUE_DAYS_WARNING   = 0        # any overdue is at least WARNING
    MAX_OVERDUE_DAYS_BLOCK     = 180      # 6 months overdue → lending BLOCKED
    BASE_PENALTY_EUR           = 1_200    # minimum ICAC fine
    DAILY_PENALTY_EUR          = 300      # accruing daily fine estimate

    @classmethod
    def compute_deadline(cls, fiscal_year_end: date) -> date:
        """Return the legal deposit deadline for given fiscal year-end."""
        # Add 7 months (standard path: 6m GM + 1m deposit)
        month = fiscal_year_end.month + cls.STANDARD_MONTHS_TO_DEPOSIT
        year  = fiscal_year_end.year + (month - 1) // 12
        month = (month - 1) % 12 + 1
        # Last day of that month
        if month == 12:
            last_day = date(year, 12, 31)
        else:
            last_day = date(year, month + 1, 1) - timedelta(days=1)
        return last_day

    @classmethod
    def assess(
        cls,
        fiscal_year_end: date,
        deposit_date: Optional[date],
        reference_date: Optional[date] = None,
    ) -> AnnualAccountsCheck:
        ref = reference_date or date.today()
        deadline = cls.compute_deadline(fiscal_year_end)

        if deposit_date:
            days_overdue = max(0, (deposit_date - deadline).days)
            if days_overdue == 0:
                status = ComplianceStatus.COMPLIANT
                penalty = 0.0
                notes = "Accounts deposited within legal deadline."
            else:
                penalty = cls.BASE_PENALTY_EUR + days_overdue * cls.DAILY_PENALTY_EUR
                if days_overdue >= cls.MAX_OVERDUE_DAYS_BLOCK:
                    status = ComplianceStatus.BLOCKED
                    notes = f"Accounts deposited {days_overdue}d late. Lending BLOCKED."
                else:
                    status = ComplianceStatus.NON_COMPLIANT
                    notes = f"Accounts deposited {days_overdue}d after deadline."
        else:
            # Not yet deposited
            days_overdue = max(0, (ref - deadline).days)
            if days_overdue == 0:
                status = ComplianceStatus.COMPLIANT
                penalty = 0.0
                notes = "Deposit deadline not yet reached."
            elif days_overdue < 30:
                status = ComplianceStatus.WARNING
                penalty = cls.BASE_PENALTY_EUR
                notes = f"Deposit overdue by {days_overdue} days."
            elif days_overdue >= cls.MAX_OVERDUE_DAYS_BLOCK:
                status = ComplianceStatus.BLOCKED
                penalty = cls.BASE_PENALTY_EUR + days_overdue * cls.DAILY_PENALTY_EUR
                notes = f"Accounts {days_overdue}d overdue. Lending BLOCKED by regulation."
            else:
                status = ComplianceStatus.NON_COMPLIANT
                penalty = cls.BASE_PENALTY_EUR + days_overdue * cls.DAILY_PENALTY_EUR
                notes = f"Accounts {days_overdue}d overdue. Corrective action required."

        return AnnualAccountsCheck(
            fiscal_year_end=fiscal_year_end,
            deposit_deadline=deadline,
            deposit_date=deposit_date,
            days_overdue=days_overdue,
            status=status,
            penalty_risk_eur=round(penalty, 2),
            notes=notes,
        )


# ---------------------------------------------------------------------------
# Litigation Risk Scorer
# ---------------------------------------------------------------------------

class LitigationRiskScorer:
    """
    Scores SME litigation risk from financial and behavioural signals.

    Factors weighted by empirical SME default study (Bank of Spain 2023):
      - payment_delay_days   : weighted 0.30
      - creditor_disputes    : weighted 0.25
      - court_proceedings    : weighted 0.25
      - debt_to_equity       : weighted 0.10
      - revenue_decline_pct  : weighted 0.10
    """

    WEIGHTS = {
        "payment_delay_days":  0.30,
        "creditor_disputes":   0.25,
        "court_proceedings":   0.25,
        "debt_to_equity":      0.10,
        "revenue_decline_pct": 0.10,
    }

    THRESHOLDS = {
        LitigationRisk.LOW:      (0.00, 0.30),
        LitigationRisk.MEDIUM:   (0.30, 0.55),
        LitigationRisk.HIGH:     (0.55, 0.80),
        LitigationRisk.CRITICAL: (0.80, 1.00),
    }

    RECOMMENDATIONS = {
        LitigationRisk.LOW:      "Standard due diligence. Proceed with normal credit assessment.",
        LitigationRisk.MEDIUM:   "Enhanced due diligence. Request supporting documentation.",
        LitigationRisk.HIGH:     "Risk committee review required. Apply stricter covenants.",
        LitigationRisk.CRITICAL: "Lending suspended. Refer to recovery department.",
    }

    @classmethod
    def score(
        cls,
        payment_delay_days: float = 0,    # avg days beyond invoice due date
        creditor_disputes: int = 0,       # number of active creditor disputes
        court_proceedings: int = 0,       # active court proceedings count
        debt_to_equity: float = 0.5,      # D/E ratio
        revenue_decline_pct: float = 0.0, # YoY revenue decline (0–100)
    ) -> LitigationRiskAssessment:

        # Normalise each factor to [0, 1]
        norm_factors = {
            "payment_delay_days":  min(payment_delay_days / 180, 1.0),
            "creditor_disputes":   min(creditor_disputes / 10, 1.0),
            "court_proceedings":   min(court_proceedings / 5, 1.0),
            "debt_to_equity":      min(max(debt_to_equity - 1.0, 0) / 4.0, 1.0),
            "revenue_decline_pct": min(max(revenue_decline_pct, 0) / 50, 1.0),
        }

        raw_score = sum(
            norm_factors[k] * w for k, w in cls.WEIGHTS.items()
        )
        score = min(1.0, raw_score)

        # Determine risk level
        risk_level = LitigationRisk.LOW
        for level, (lo, hi) in cls.THRESHOLDS.items():
            if lo <= score <= hi:
                risk_level = level
                break

        # Build flags
        flags: List[str] = []
        if payment_delay_days > 90:
            flags.append(f"PAYMENT_DELAY_{int(payment_delay_days)}D")
        if creditor_disputes > 3:
            flags.append(f"MULTIPLE_CREDITOR_DISPUTES_{creditor_disputes}")
        if court_proceedings > 0:
            flags.append(f"ACTIVE_COURT_PROCEEDINGS_{court_proceedings}")
        if debt_to_equity > 3.0:
            flags.append(f"HIGH_LEVERAGE_DTE_{debt_to_equity:.1f}")
        if revenue_decline_pct > 20:
            flags.append(f"REVENUE_DECLINE_{revenue_decline_pct:.0f}pct")

        return LitigationRiskAssessment(
            risk_level=risk_level,
            risk_score=round(score, 4),
            flags=flags,
            contributing_factors={k: round(v, 4) for k, v in norm_factors.items()},
            recommendation=cls.RECOMMENDATIONS[risk_level],
        )


# ---------------------------------------------------------------------------
# AML / KYB Screener (simplified — production would call external APIs)
# ---------------------------------------------------------------------------

class AMLScreener:
    """
    Anti-Money Laundering / Know-Your-Business screening.

    In production: integrates with ComplyAdvantage, Refinitiv WC, or SEON.
    This implementation uses a local watchlist for demonstration purposes.
    """

    # Simulated sanctioned-entity fragments (real deployment uses live feeds)
    _WATCHLIST_FRAGMENTS = [
        "offshore_holding", "cayman_capital", "panama_invest",
        "shell_corp", "anonymous_ltd", "undisclosed",
    ]

    @classmethod
    def screen(cls, entity_name: str, jurisdiction: str = "ES") -> AMLScreeningResult:
        name_lower = entity_name.lower()
        matched: List[str] = []
        confidence = 0.0

        for fragment in cls._WATCHLIST_FRAGMENTS:
            if fragment.replace("_", " ") in name_lower:
                matched.append(f"INTERNAL_WATCHLIST:{fragment.upper()}")
                confidence = max(confidence, 0.92)

        # High-risk jurisdiction check
        HIGH_RISK_JURISDICTIONS = {"KP", "IR", "CU", "SY", "VE", "MM"}
        if jurisdiction.upper() in HIGH_RISK_JURISDICTIONS:
            matched.append(f"HIGH_RISK_JURISDICTION:{jurisdiction}")
            confidence = max(confidence, 0.80)

        if not matched:
            flag = AMLFlag.CLEAR
            confidence = 0.02
            details = "No matches on watchlists or high-risk jurisdiction lists."
        elif confidence >= 0.85:
            flag = AMLFlag.BLOCKED
            details = f"Entity matched {len(matched)} AML watchlist entries."
        else:
            flag = AMLFlag.WATCHLIST
            details = f"Entity flagged for review. Matched: {matched}"

        return AMLScreeningResult(
            entity_name=entity_name,
            flag=flag,
            matched_lists=matched,
            confidence=round(confidence, 4),
            details=details,
        )


# ---------------------------------------------------------------------------
# Main Compliance Engine
# ---------------------------------------------------------------------------

class RegulatoryComplianceEngine:
    """
    Orchestrates all compliance checks for a single SME entity.

    Usage
    -----
    >>> engine = RegulatoryComplianceEngine()
    >>> report = engine.run_full_compliance_check(
    ...     entity_name="Acme Logistics SL",
    ...     cif="B12345678",
    ...     fiscal_year_end=date(2023, 12, 31),
    ...     deposit_date=None,
    ...     payment_delay_days=45,
    ...     creditor_disputes=1,
    ...     court_proceedings=0,
    ...     debt_to_equity=1.8,
    ...     revenue_decline_pct=5.0,
    ... )
    >>> print(report.overall_status)
    """

    def __init__(self) -> None:
        self._registry = MercantileRegistryRules()
        self._litigation = LitigationRiskScorer()
        self._aml = AMLScreener()
        logger.info("RegulatoryComplianceEngine initialised.")

    def run_full_compliance_check(
        self,
        entity_name: str,
        cif: str,
        fiscal_year_end: date,
        deposit_date: Optional[date] = None,
        payment_delay_days: float = 0,
        creditor_disputes: int = 0,
        court_proceedings: int = 0,
        debt_to_equity: float = 0.5,
        revenue_decline_pct: float = 0.0,
        jurisdiction: str = "ES",
        reference_date: Optional[date] = None,
    ) -> ComplianceReport:
        """Run all compliance sub-checks and produce a consolidated report."""

        ref = reference_date or date.today()
        report_id = f"RCE-{uuid.uuid4().hex[:8].upper()}"

        # --- Sub-checks ---
        accounts_check = MercantileRegistryRules.assess(
            fiscal_year_end, deposit_date, ref
        )
        litigation_check = LitigationRiskScorer.score(
            payment_delay_days, creditor_disputes,
            court_proceedings, debt_to_equity, revenue_decline_pct,
        )
        aml_check = AMLScreener.screen(entity_name, jurisdiction)

        # --- Aggregate status ---
        blocking_reasons: List[str] = []

        if accounts_check.status == ComplianceStatus.BLOCKED:
            blocking_reasons.append(accounts_check.notes)
        if litigation_check.risk_level == LitigationRisk.CRITICAL:
            blocking_reasons.append("CRITICAL litigation risk — lending suspended.")
        if aml_check.flag == AMLFlag.BLOCKED:
            blocking_reasons.append("AML screening: entity BLOCKED on watchlist.")

        lending_eligible = len(blocking_reasons) == 0

        # Overall status hierarchy: BLOCKED > NON_COMPLIANT > WARNING > COMPLIANT
        statuses = [accounts_check.status]
        if litigation_check.risk_level in (LitigationRisk.HIGH, LitigationRisk.CRITICAL):
            statuses.append(ComplianceStatus.NON_COMPLIANT)
        if aml_check.flag == AMLFlag.BLOCKED:
            statuses.append(ComplianceStatus.BLOCKED)
        elif aml_check.flag == AMLFlag.WATCHLIST:
            statuses.append(ComplianceStatus.WARNING)

        hierarchy = [
            ComplianceStatus.BLOCKED,
            ComplianceStatus.NON_COMPLIANT,
            ComplianceStatus.WARNING,
            ComplianceStatus.COMPLIANT,
        ]
        overall = ComplianceStatus.COMPLIANT
        for h in hierarchy:
            if h in statuses:
                overall = h
                break

        # Composite compliance score (0–100, higher = better)
        accounts_score = {
            ComplianceStatus.COMPLIANT:     40,
            ComplianceStatus.WARNING:       25,
            ComplianceStatus.NON_COMPLIANT: 10,
            ComplianceStatus.BLOCKED:       0,
        }[accounts_check.status]
        litigation_score = int((1 - litigation_check.risk_score) * 40)
        aml_score = {AMLFlag.CLEAR: 20, AMLFlag.WATCHLIST: 8, AMLFlag.BLOCKED: 0}[aml_check.flag]
        composite_score = accounts_score + litigation_score + aml_score

        next_review = (ref + timedelta(days=90)).isoformat()

        return ComplianceReport(
            report_id=report_id,
            entity_name=entity_name,
            cif=cif,
            generated_at=datetime.utcnow().isoformat(),
            overall_status=overall,
            annual_accounts=accounts_check,
            litigation=litigation_check,
            aml_screening=aml_check,
            lending_eligible=lending_eligible,
            blocking_reasons=blocking_reasons,
            compliance_score=composite_score,
            next_review_date=next_review,
            regulatory_references=[
                "LSC Art. 279 — Accounts Deposit",
                "Ley 22/2003 — Insolvency",
                "6AMLD — AML Directive",
                "Bank of Spain Circular 4/2017",
            ],
        )

    def check_accounts_deadline(
        self, fiscal_year_end: date, deposit_date: Optional[date] = None
    ) -> AnnualAccountsCheck:
        """Lightweight check: annual accounts only."""
        return MercantileRegistryRules.assess(fiscal_year_end, deposit_date)

    def screen_entity(self, entity_name: str, jurisdiction: str = "ES") -> AMLScreeningResult:
        """Lightweight AML / KYB screening."""
        return AMLScreener.screen(entity_name, jurisdiction)


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    engine = RegulatoryComplianceEngine()

    report = engine.run_full_compliance_check(
        entity_name="Acme Logistics SL",
        cif="B12345678",
        fiscal_year_end=date(2023, 12, 31),
        deposit_date=None,
        payment_delay_days=120,
        creditor_disputes=4,
        court_proceedings=1,
        debt_to_equity=3.5,
        revenue_decline_pct=25.0,
    )

    import dataclasses
    print(json.dumps(dataclasses.asdict(report), indent=2, default=str))
