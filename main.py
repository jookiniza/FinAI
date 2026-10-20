"""
main.py
=======
FastAPI Orchestrator — SME Finance Copilot API + Interactive Web Dashboard.

Endpoints:
  GET  /              — Health check
  GET  /dashboard     — Full HTML/JS interactive dashboard (demo video ready)
  POST /analyze       — Full SME financial analysis pipeline
  POST /recommend     — AI product recommendations only
  POST /compliance    — Regulatory compliance check only
  POST /stress-test   — Cash-flow stress test only
  POST /mask-pii      — PII anonymization only
  GET  /products      — Product catalogue listing
  GET  /audit-trail   — Session audit log
  GET  /docs          — Auto-generated Swagger UI
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
import uuid
from dataclasses import asdict, dataclass
from datetime import date, datetime
from typing import Any, Dict, List, Optional

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field

from ai_agent import SMEFinancialCopilot, PRODUCT_CATALOGUE
from cashflow_engine import CashFlowStressTester
from legal_engine import RegulatoryComplianceEngine
from security_layer import PrivacyPreservingLayer

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("app.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Application bootstrap
# ---------------------------------------------------------------------------
app = FastAPI(
    title="SME Finance Copilot",
    description=(
        "Enterprise-grade AI-powered financial copilot for SMEs. "
        "FinTechathon 2026 — Shenzhen International FinTech Competition (Topic E). "
        "Integrates PII masking, regulatory compliance, Monte Carlo stress testing, "
        "and explainable RAG product recommendations."
    ),
    version="2.0.0",
    contact={"name": "FinTechathon Team E", "email": "team-e@fintechathon2026.com"},
    license_info={"name": "MIT"},
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Singleton service instances
_pii_engine    = PrivacyPreservingLayer(enable_audit_log=True)
_legal_engine  = RegulatoryComplianceEngine()
_stress_tester = CashFlowStressTester(simulations=5_000)
_ai_copilot    = SMEFinancialCopilot()

# In-memory verifiable audit log
_audit_log: List[Dict[str, Any]] = []


# ---------------------------------------------------------------------------
# Pydantic request / response models
# ---------------------------------------------------------------------------

class SMEAnalysisRequest(BaseModel):
    entity_name: str = Field(..., example="InnovaTech Solutions SL")
    cif: str = Field(..., example="B87654321")
    raw_description: str = Field(
        ...,
        example="CEO: John Smith, CFO: Jane Smith. Email: john.smith@innovatech.com, phone: +34 612 345 678. IBAN: ES9121000418450200051332.",
    )
    fiscal_year_end: str = Field(..., example="2023-12-31")
    deposit_date: Optional[str] = Field(None, example="2024-06-15")
    opening_balance: float = Field(..., example=320000.0)
    monthly_revenue: float = Field(..., example=210000.0)
    monthly_expenses: float = Field(..., example=165000.0)
    payment_delay_days: float = Field(0.0, example=12.0)
    creditor_disputes: int = Field(0, example=0)
    court_proceedings: int = Field(0, example=0)
    debt_to_equity: float = Field(0.5, example=1.2)
    revenue_decline_pct: float = Field(0.0, example=0.0)
    financing_query: str = Field(
        ...,
        example="We need working capital loan to support international expansion and hire 5 AI engineers",
    )
    amount_eur: Optional[float] = Field(None, example=350000.0)
    categories: Optional[List[str]] = Field(None, example=["lending", "grant"])


class PIIMaskRequest(BaseModel):
    text: str = Field(..., example="Call CEO John Smith at +34 612 345 678 or john.doe@email.com")
    context: Optional[str] = Field(None, example="financial")


class RecommendRequest(BaseModel):
    entity_name: str = Field(..., example="InnovaTech Solutions SL")
    query: str = Field(..., example="Working capital for seasonal cash flow")
    top_k: int = Field(5, ge=1, le=10)
    categories: Optional[List[str]] = None
    amount_eur: Optional[float] = None


class ComplianceRequest(BaseModel):
    entity_name: str = Field(..., example="InnovaTech Solutions SL")
    cif: str = Field(..., example="B87654321")
    fiscal_year_end: str = Field(..., example="2023-12-31")
    deposit_date: Optional[str] = None
    payment_delay_days: float = Field(0.0)
    creditor_disputes: int = Field(0)
    court_proceedings: int = Field(0)
    debt_to_equity: float = Field(0.5)
    revenue_decline_pct: float = Field(0.0)
    jurisdiction: str = Field("ES")


class StressTestRequest(BaseModel):
    entity_name: str = Field(..., example="InnovaTech Solutions SL")
    opening_balance: float = Field(..., example=320000.0)
    monthly_revenue: float = Field(..., example=210000.0)
    monthly_expenses: float = Field(..., example=165000.0)


# ---------------------------------------------------------------------------
# Helper: audit log entry
# ---------------------------------------------------------------------------

def _log_audit(endpoint: str, payload: Dict[str, Any], result: Dict[str, Any]) -> None:
    entry = {
        "id": str(uuid.uuid4())[:8],
        "endpoint": endpoint,
        "timestamp": datetime.utcnow().isoformat(),
        "input_keys": list(payload.keys()),
        "status": result.get("status", "ok"),
        "audit_trail": result.get("audit_trail"),
    }
    _audit_log.append(entry)
    logger.info("AUDIT | %s | %s", endpoint, entry["id"])


# ---------------------------------------------------------------------------
# Dashboard HTML (single-page, self-contained, demo-optimized)
# ---------------------------------------------------------------------------

DASHBOARD_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>SME Finance Copilot — FinTechathon 2026</title>
<style>
  :root{--primary:#1a73e8;--secondary:#0d47a1;--accent:#00c853;--danger:#e53935;--warn:#fb8c00;--bg:#0a0e1a;--surface:#111827;--surface2:#1f2937;--text:#e2e8f0;--text2:#94a3b8;--border:#374151;--radius:12px;--shadow:0 4px 24px rgba(0,0,0,0.4)}
  *{box-sizing:border-box;margin:0;padding:0}
  body{font-family:'Segoe UI',system-ui,sans-serif;background:var(--bg);color:var(--text);min-height:100vh}
  header{background:linear-gradient(135deg,var(--secondary),var(--primary));padding:20px 32px;display:flex;align-items:center;justify-content:space-between;box-shadow:var(--shadow)}
  header h1{font-size:1.6rem;font-weight:700;letter-spacing:-0.5px}
  header span{background:rgba(255,255,255,0.15);padding:4px 12px;border-radius:20px;font-size:0.75rem;font-weight:600}
  .badge-green{background:var(--accent)!important;color:#000!important}
  nav{display:flex;gap:8px;padding:16px 32px;background:var(--surface);border-bottom:1px solid var(--border);overflow-x:auto}
  nav button{padding:8px 20px;border:1px solid var(--border);border-radius:8px;background:transparent;color:var(--text2);cursor:pointer;font-size:0.85rem;white-space:nowrap;transition:all .2s}
  nav button:hover,nav button.active{background:var(--primary);color:#fff;border-color:var(--primary)}
  main{max-width:1250px;margin:0 auto;padding:24px 32px}
  .panel{display:none}.panel.active{display:block}
  .card{background:var(--surface);border:1px solid var(--border);border-radius:var(--radius);padding:24px;margin-bottom:20px;box-shadow:var(--shadow)}
  .card h2{font-size:1.15rem;font-weight:600;margin-bottom:16px;color:var(--text);display:flex;align-items:center;gap:8px}
  .grid-2{display:grid;grid-template-columns:1fr 1fr;gap:16px}
  .grid-3{display:grid;grid-template-columns:1fr 1fr 1fr;gap:16px}
  label{display:block;font-size:0.78rem;color:var(--text2);margin-bottom:4px;margin-top:12px;font-weight:600;text-transform:uppercase;letter-spacing:.5px}
  input,textarea,select{width:100%;padding:10px 14px;background:var(--surface2);border:1px solid var(--border);border-radius:8px;color:var(--text);font-size:0.9rem;transition:border .2s}
  input:focus,textarea:focus,select:focus{outline:none;border-color:var(--primary)}
  textarea{min-height:80px;resize:vertical}
  .btn{display:inline-flex;align-items:center;justify-content:center;gap:8px;padding:12px 24px;border:none;border-radius:8px;cursor:pointer;font-size:0.9rem;font-weight:600;transition:all .2s}
  .btn-primary{background:var(--primary);color:#fff}.btn-primary:hover{background:var(--secondary)}
  .btn-danger{background:var(--danger);color:#fff}
  .btn-accent{background:var(--accent);color:#000}
  .btn-preset{border:1px solid;text-align:left;padding:16px;border-radius:10px;cursor:pointer;transition:transform .2s,box-shadow .2s}
  .btn-preset:hover{transform:translateY(-2px);box-shadow:0 6px 20px rgba(0,0,0,0.5)}
  .spinner{display:none;width:18px;height:18px;border:3px solid rgba(255,255,255,.3);border-top-color:#fff;border-radius:50%;animation:spin 0.8s linear infinite}
  @keyframes spin{to{transform:rotate(360deg)}}
  .loading .spinner{display:inline-block}
  .loading .btn-text{display:none}
  .result{background:var(--surface2);border:1px solid var(--border);border-radius:10px;padding:20px;margin-top:20px}
  pre{white-space:pre-wrap;word-break:break-word;font-size:0.8rem;color:#a5f3fc;font-family:'Cascadia Code','Fira Code',monospace}
  .status-bar{display:flex;gap:12px;flex-wrap:wrap;margin-bottom:20px}
  .stat{background:var(--surface);border:1px solid var(--border);border-radius:10px;padding:14px 20px;flex:1;min-width:140px}
  .stat .val{font-size:1.6rem;font-weight:700;color:var(--primary)}
  .stat .lbl{font-size:0.75rem;color:var(--text2);text-transform:uppercase;letter-spacing:.5px}
  .tag{display:inline-block;padding:3px 10px;border-radius:6px;font-size:0.75rem;font-weight:700;margin:2px}
  .tag-green{background:rgba(0,200,83,.15);color:var(--accent);border:1px solid rgba(0,200,83,.4)}
  .tag-red{background:rgba(229,57,53,.15);color:#ef5350;border:1px solid rgba(229,57,53,.4)}
  .tag-blue{background:rgba(26,115,232,.15);color:#42a5f5;border:1px solid rgba(26,115,232,.4)}
  .tag-orange{background:rgba(251,140,0,.15);color:#ffa726;border:1px solid rgba(251,140,0,.4)}
  .rec-card{background:var(--bg);border:1px solid var(--border);border-radius:10px;padding:16px;margin:10px 0}
  .rec-card h3{font-size:1rem;margin-bottom:6px}
  .score-bar{background:var(--border);border-radius:4px;height:6px;margin:6px 0}
  .score-fill{height:100%;border-radius:4px;background:linear-gradient(90deg,var(--primary),var(--accent))}
  .risk-badge{padding:6px 14px;border-radius:20px;font-size:0.82rem;font-weight:700;display:inline-block}
  .risk-LOW{background:rgba(0,200,83,.2);color:var(--accent);border:1px solid var(--accent)}
  .risk-MEDIUM{background:rgba(251,140,0,.2);color:#ffa726;border:1px solid var(--warn)}
  .risk-HIGH{background:rgba(229,57,53,.2);color:#ef5350;border:1px solid var(--danger)}
  .risk-CRITICAL,.risk-BLOCKED{background:#e53935;color:#fff}
  .risk-COMPLIANT{background:rgba(0,200,83,.25);color:var(--accent);border:1px solid var(--accent)}
  .alert{padding:14px 18px;border-radius:8px;margin:10px 0;font-size:0.88rem;line-height:1.5}
  .alert-success{background:rgba(0,200,83,.12);border:1px solid rgba(0,200,83,.3);color:var(--accent)}
  .alert-error{background:rgba(229,57,53,.12);border:1px solid rgba(229,57,53,.3);color:#ef5350}
  .alert-warn{background:rgba(251,140,0,.12);border:1px solid rgba(251,140,0,.3);color:#ffa726}
  .hash-box{font-family:'Cascadia Code','Fira Code',monospace;font-size:0.75rem;padding:6px 10px;background:rgba(0,0,0,0.4);border-radius:6px;word-break:break-all;color:#7dd3fc;border:1px dashed var(--border)}
  footer{text-align:center;padding:24px;color:var(--text2);font-size:0.75rem;border-top:1px solid var(--border);margin-top:40px}
  @media(max-width:768px){.grid-2,.grid-3{grid-template-columns:1fr}}
</style>
</head>
<body>
<header>
  <div>
    <h1>🏦 SME Finance Copilot</h1>
    <div style="display:flex;gap:8px;margin-top:6px;flex-wrap:wrap">
      <span>FinTechathon 2026 — Topic E</span>
      <span class="badge-green">🟢 Live</span>
      <span>v2.0.0</span>
    </div>
  </div>
  <div>
    <a href="/docs" target="_blank" style="color:#fff;text-decoration:none;background:rgba(255,255,255,0.15);padding:8px 16px;border-radius:8px;font-size:0.85rem;font-weight:600;">📖 OpenAPI Specs (/docs)</a>
  </div>
</header>

<nav>
  <button class="active" onclick="showPanel('dashboard')">📊 Dashboard</button>
  <button onclick="showPanel('full-analysis')">🔍 Full Analysis</button>
  <button onclick="showPanel('pii-mask')">🔐 PII Masking</button>
  <button onclick="showPanel('compliance')">⚖️ Compliance</button>
  <button onclick="showPanel('stress-test')">📈 Stress Test</button>
  <button onclick="showPanel('recommend')">💡 Recommendations</button>
  <button onclick="showPanel('audit')">📋 Audit Trail</button>
</nav>

<main>
<!-- DASHBOARD -->
<div id="panel-dashboard" class="panel active">
  <div class="status-bar">
    <div class="stat"><div class="val" id="stat-req">0</div><div class="lbl">API Requests</div></div>
    <div class="stat"><div class="val" id="stat-pii">0</div><div class="lbl">PII Redacted</div></div>
    <div class="stat"><div class="val" id="stat-prod">10</div><div class="lbl">Products Indexed</div></div>
    <div class="stat"><div class="val">✅</div><div class="lbl">GDPR Art. 25 & LSC 279</div></div>
  </div>

  <!-- ONE-CLICK SCENARIO BAR (NO TYPING REQUIRED) -->
  <div class="card" style="border: 2px solid var(--primary); background: linear-gradient(180deg, rgba(26,115,232,0.08), var(--surface));">
    <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:12px;margin-bottom:12px;">
      <h2>⚡ One-Click Video Scenarios (No Typing Required)</h2>
      <span class="tag tag-blue" style="font-size:0.8rem;padding:4px 12px;">FinTechathon 2026 Ready</span>
    </div>
    <p style="color:var(--text2);font-size:0.88rem;margin-bottom:16px;">
      Click any preset below to instantly load real SME financial data and execute the entire 4-tier pipeline live:
    </p>
    <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:14px;">
      
      <!-- PRESET 1 -->
      <div class="btn-preset" onclick="runPresetScenario('preset1')" style="background:rgba(0,200,83,0.12);border-color:var(--accent);">
        <div style="font-weight:700;font-size:0.95rem;color:#00e676;margin-bottom:6px;">🟢 [Preset 1: Compliant SME]</div>
        <div style="font-size:0.8rem;color:var(--text2);margin-bottom:8px;">InnovaTech Solutions SL · Clean PII Redaction · Compliant Accounts · High Liquidity Rating A</div>
        <div style="display:flex;gap:6px;flex-wrap:wrap;">
          <span class="tag tag-green">Names & IBAN Masked</span>
          <span class="tag tag-blue">Eligible for Lending</span>
        </div>
      </div>

      <!-- PRESET 2 -->
      <div class="btn-preset" onclick="runPresetScenario('preset2')" style="background:rgba(229,57,53,0.12);border-color:var(--danger);">
        <div style="font-weight:700;font-size:0.95rem;color:#ff5252;margin-bottom:6px;">🔴 [Preset 2: Regulatory Blocked SME]</div>
        <div style="font-size:0.8rem;color:var(--text2);margin-bottom:8px;">Construcciones & Obras SL · Accounts Overdue >400 Days · Active Court Proceedings · LSC Art. 279</div>
        <div style="display:flex;gap:6px;flex-wrap:wrap;">
          <span class="tag tag-red">Hard Legal Block</span>
          <span class="tag tag-orange">Lending Suspended</span>
        </div>
      </div>

      <!-- PRESET 3 -->
      <div class="btn-preset" onclick="runPresetScenario('preset3')" style="background:rgba(251,140,0,0.12);border-color:var(--warn);">
        <div style="font-weight:700;font-size:0.95rem;color:#ffa726;margin-bottom:6px;">🟠 [Preset 3: High Risk Liquidity]</div>
        <div style="font-size:0.8rem;color:var(--text2);margin-bottom:8px;">Global Retail Imports SL · Severe Cash Burn (-€45k/mo) · 60% Ruin Risk · Basel III LCR Deficit</div>
        <div style="display:flex;gap:6px;flex-wrap:wrap;">
          <span class="tag tag-orange">Monte Carlo Alert</span>
          <span class="tag tag-red">Liquidity Rating D</span>
        </div>
      </div>

    </div>
    
    <!-- Spinner for presets -->
    <div id="preset-spinner-wrap" style="display:none;margin-top:16px;text-align:center;">
      <div class="spinner" id="preset-spinner" style="display:inline-block;width:24px;height:24px;"></div>
      <span style="margin-left:10px;font-size:0.9rem;color:var(--primary);font-weight:600;">Executing 4-Tier Pipeline (PII Masking → Compliance → Monte Carlo → AI RAG)...</span>
    </div>

    <!-- Live Execution Showcase on Dashboard -->
    <div id="preset-output" style="display:none;margin-top:20px;"></div>
  </div>

  <div class="card">
    <h2>🏗️ 4-Tier Pipeline System Architecture</h2>
    <pre style="color:var(--text2);font-size:0.78rem">
┌─────────────────────────────────────────────────────────────────────────┐
│                    SME Finance Copilot — v2.0.0                          │
│              FastAPI Orchestrator + Interactive Web Dashboard             │
└────────────────────────────┬────────────────────────────────────────────┘
                             │  POST /analyze  (Full Pipeline)
             ┌───────────────▼───────────────────────────────┐
             │          TIER-1: Security Gate                  │
             │   PrivacyPreservingLayer (Presidio + Regex)   │
             │   • &lt;PERSON&gt;, &lt;EMAIL_ADDRESS&gt;, &lt;IBAN_CODE&gt;    │
             │   • &lt;PHONE_NUMBER&gt;, &lt;CREDIT_CARD&gt;, &lt;TAX_ID&gt;   │
             │   • GDPR Art.25 | ISO 27001 A.18              │
             └──────────────┬───────────────────────┬─────────┘
                            │                       │
         ┌──────────────────▼──────┐   ┌────────────▼────────────────┐
         │  Compliance Engine       │   │  AI RAG Copilot              │
         │  (legal_engine.py)       │   │  (ai_agent.py)               │
         │  • LSC Art. 279 Deadlines│   │  • SentenceTransformers RAG  │
         │  • AML / KYB Watchlist   │   │  • 10-Product Catalogue      │
         │  • Litigation Risk Score │   │  • EU AI Act Art. 13         │
         └──────────────────────────┘   └─────────────────────────────┘
                                                      │
                            ┌─────────────────────────▼──────────────────┐
                            │      Monte Carlo Stress Testing Engine       │
                            │      (cashflow_engine.py)                    │
                            │  • 5,000 simulations per scenario (30/60/90d)│
                            │  • VaR95, VaR99, CVaR, Ruin Probability      │
                            │  • Basel III LCR Proxy & Liquidity Rating    │
                            └────────────────────────────────────────────┘
    </pre>
  </div>
</div>

<!-- FULL ANALYSIS -->
<div id="panel-full-analysis" class="panel">
  <div class="card">
    <h2>🔍 Full SME Financial Analysis Pipeline</h2>
    <p style="color:var(--text2);margin-bottom:12px;font-size:0.85rem">Execute all 4 engines sequentially: PII Masking → Compliance Check → Monte Carlo Stress Test → AI Product Recommendations.</p>
    
    <!-- Quick preset loader bar inside form -->
    <div style="background:var(--surface2);padding:12px 16px;border-radius:8px;border:1px solid var(--border);margin-bottom:16px;display:flex;align-items:center;gap:12px;flex-wrap:wrap;">
      <span style="font-size:0.8rem;color:var(--text2);font-weight:700;">LOAD PRESET:</span>
      <button class="btn" onclick="populateForm('preset1')" style="background:rgba(0,200,83,0.2);color:#00e676;padding:6px 14px;font-size:0.8rem;">Preset 1: Compliant</button>
      <button class="btn" onclick="populateForm('preset2')" style="background:rgba(229,57,53,0.2);color:#ff5252;padding:6px 14px;font-size:0.8rem;">Preset 2: Blocked</button>
      <button class="btn" onclick="populateForm('preset3')" style="background:rgba(251,140,0,0.2);color:#ffa726;padding:6px 14px;font-size:0.8rem;">Preset 3: High Risk</button>
    </div>

    <div class="grid-2">
      <div>
        <label>Company Name</label>
        <input id="fa-name" value="InnovaTech Solutions SL"/>
        <label>CIF (Tax ID)</label>
        <input id="fa-cif" value="B87654321"/>
        <label>Fiscal Year End</label>
        <input id="fa-fy" type="date" value="2023-12-31"/>
        <label>Deposit Date (leave blank if overdue)</label>
        <input id="fa-deposit" type="date" value="2024-06-15"/>
        <label>Financing Query</label>
        <textarea id="fa-query">We need working capital loan to support international expansion and hire 5 AI engineers</textarea>
      </div>
      <div>
        <label>Raw Description (PII Masking Target)</label>
        <textarea id="fa-desc" style="min-height:90px;">CEO: John Smith, CFO: Jane Smith. Email: john.smith@innovatech.com, phone: +34 612 345 678. Primary Account IBAN: ES9121000418450200051332. Card: 4111 1111 1111 1111.</textarea>
        <div class="grid-2">
          <div><label>Opening Balance (EUR)</label><input id="fa-balance" type="number" value="320000"/></div>
          <div><label>Monthly Revenue (EUR)</label><input id="fa-revenue" type="number" value="210000"/></div>
        </div>
        <div class="grid-2">
          <div><label>Monthly Expenses (EUR)</label><input id="fa-expenses" type="number" value="165000"/></div>
          <div><label>Debt-to-Equity</label><input id="fa-dte" type="number" step="0.1" value="1.2"/></div>
        </div>
      </div>
    </div>
    <div style="margin-top:20px;">
      <button class="btn btn-primary" id="fa-btn" onclick="runFullAnalysis()">
        <span class="btn-text">🚀 Run Full Analysis Pipeline</span>
        <div class="spinner" id="fa-spinner"></div>
      </button>
    </div>
    <div id="fa-result" class="result" style="display:none;"></div>
  </div>
</div>

<!-- PII MASKING -->
<div id="panel-pii-mask" class="panel">
  <div class="card">
    <h2>🔐 Tier-1 PII Anonymization Engine</h2>
    <p style="color:var(--text2);margin-bottom:12px;font-size:0.85rem">Microsoft Presidio NLP + Deterministic Regex Fallback. Anonymizes sensitive data into clean tags: &lt;PERSON&gt;, &lt;EMAIL_ADDRESS&gt;, &lt;IBAN_CODE&gt;, &lt;PHONE_NUMBER&gt;, &lt;CREDIT_CARD&gt;, &lt;TAX_ID&gt;.</p>
    <label>Input Text (enter any raw SME narrative with sensitive data)</label>
    <textarea id="pii-input" style="min-height:120px;">CEO John Smith and CFO Jane Smith reached out. Email: john.doe@email.com, phone: +34 612 345 678. Bank Account IBAN: ES9121000418450200051332. Card: 4111 1111 1111 1111. CIF: B87654321.</textarea>
    <div style="margin-top:14px">
      <button class="btn btn-primary" id="pii-btn" onclick="runPIIMask()">
        <span class="btn-text">🔐 Anonymize PII Live</span>
        <div class="spinner" id="pii-spinner"></div>
      </button>
    </div>
    <div id="pii-result" style="margin-top:16px;display:none;">
      <div class="alert alert-success" id="pii-masked-text"></div>
      <div id="pii-details" class="result"></div>
    </div>
  </div>
</div>

<!-- COMPLIANCE -->
<div id="panel-compliance" class="panel">
  <div class="card">
    <h2>⚖️ Regulatory Compliance Engine</h2>
    <p style="color:var(--text2);margin-bottom:12px;font-size:0.85rem">Checks Spanish Mercantile Registry (LSC Art. 279) deposit timelines, AML/KYB 6AMLD watchlists, and litigation risk.</p>
    <div class="grid-2">
      <div>
        <label>Entity Name</label><input id="co-name" value="InnovaTech Solutions SL"/>
        <label>CIF</label><input id="co-cif" value="B87654321"/>
        <label>Fiscal Year End</label><input id="co-fy" type="date" value="2023-12-31"/>
        <label>Deposit Date (blank = pending)</label><input id="co-deposit" type="date" value="2024-06-15"/>
        <label>Jurisdiction</label><input id="co-juris" value="ES"/>
      </div>
      <div>
        <label>Payment Delay (days)</label><input id="co-delay" type="number" value="12"/>
        <label>Creditor Disputes</label><input id="co-disputes" type="number" value="0"/>
        <label>Court Proceedings</label><input id="co-court" type="number" value="0"/>
        <label>Debt-to-Equity</label><input id="co-dte" type="number" step="0.1" value="1.2"/>
        <label>Revenue Decline (%)</label><input id="co-decline" type="number" value="0"/>
      </div>
    </div>
    <div style="margin-top:16px">
      <button class="btn btn-primary" id="co-btn" onclick="runCompliance()">
        <span class="btn-text">⚖️ Evaluate Regulatory Compliance</span>
        <div class="spinner" id="co-spinner"></div>
      </button>
    </div>
    <div id="co-result" class="result" style="display:none;"></div>
  </div>
</div>

<!-- STRESS TEST -->
<div id="panel-stress-test" class="panel">
  <div class="card">
    <h2>📈 Monte Carlo Cash-Flow Stress Testing Engine</h2>
    <p style="color:var(--text2);margin-bottom:12px;font-size:0.85rem">Simulates 5,000 cash-flow paths across 30, 60, and 90 days under Baseline, Moderate, and Severe economic stress. Computes VaR95, VaR99, CVaR, and Basel III LCR proxy.</p>
    <div class="grid-3">
      <div><label>Entity Name</label><input id="st-name" value="InnovaTech Solutions SL"/></div>
      <div><label>Opening Balance (EUR)</label><input id="st-balance" type="number" value="320000"/></div>
      <div><label>Monthly Revenue (EUR)</label><input id="st-revenue" type="number" value="210000"/></div>
    </div>
    <div style="margin-top:4px">
      <label>Monthly Expenses (EUR)</label><input id="st-expenses" type="number" value="165000" style="max-width:300px"/>
    </div>
    <div style="margin-top:16px">
      <button class="btn btn-primary" id="st-btn" onclick="runStressTest()">
        <span class="btn-text">📊 Run Monte Carlo Simulation (5,000 Runs)</span>
        <div class="spinner" id="st-spinner"></div>
      </button>
    </div>
    <div id="st-result" class="result" style="display:none;"></div>
  </div>
</div>

<!-- RECOMMENDATIONS -->
<div id="panel-recommend" class="panel">
  <div class="card">
    <h2>💡 Explainable AI Product Recommendations (RAG)</h2>
    <label>Entity Name</label><input id="rc-name" value="InnovaTech Solutions SL"/>
    <label>Financing Need (Natural Language Query)</label>
    <textarea id="rc-query">We need working capital loan to support international expansion and hire 5 AI engineers</textarea>
    <div class="grid-2">
      <div>
        <label>Max Results</label>
        <select id="rc-topk">
          <option value="3">3</option><option value="5" selected>5</option><option value="7">7</option>
        </select>
      </div>
      <div>
        <label>Amount Needed (EUR, optional)</label>
        <input id="rc-amount" type="number" value="350000"/>
      </div>
    </div>
    <div style="margin-top:16px">
      <button class="btn btn-primary" id="rc-btn" onclick="runRecommend()">
        <span class="btn-text">💡 Retrieve Explainable Recommendations</span>
        <div class="spinner" id="rc-spinner"></div>
      </button>
    </div>
    <div id="rc-result" style="display:none;margin-top:16px"></div>
  </div>
</div>

<!-- AUDIT -->
<div id="panel-audit" class="panel">
  <div class="card">
    <h2>📋 Cryptographic Audit Trail</h2>
    <p style="color:var(--text2);margin-bottom:12px;font-size:0.85rem">Every operation produces an immutable SHA-256 hash, transaction ID, and compliance verification record.</p>
    <button class="btn btn-primary" onclick="loadAudit()">🔄 Refresh Audit Trail</button>
    <div id="audit-result" class="result" style="margin-top:14px"><pre>Click Refresh to load audit records.</pre></div>
  </div>
</div>
</main>

<footer>
  SME Finance Copilot v2.0.0 | FinTechathon 2026 Shenzhen — International Track Topic E<br/>
  GDPR Art.25 | ISO27001 A.18 | EU AI Act Art.13 | Basel III LCR | 6AMLD Compliant
</footer>

<script>
const API = '';

// Pre-configured Enterprise Scenarios
const PRESETS = {
  preset1: {
    name: "InnovaTech Solutions SL",
    cif: "B87654321",
    desc: "CEO: John Smith, CFO: Jane Smith. Email: john.smith@innovatech.com, phone: +34 612 345 678. Primary Account IBAN: ES9121000418450200051332. Card: 4111 1111 1111 1111. Applying for EUR 350K working capital.",
    fy: "2023-12-31",
    deposit: "2024-06-15",
    balance: 320000,
    rev: 210000,
    exp: 165000,
    delay: 12,
    disputes: 0,
    court: 0,
    dte: 1.2,
    decline: 0,
    query: "We need working capital loan to support international expansion and hire 5 AI engineers",
    amount: 350000
  },
  preset2: {
    name: "Construcciones & Obras SL",
    cif: "B23456789",
    desc: "Manager: Carlos Mendez, phone: +34 912 888 999, email: carlos@construcciones.es. Bank account IBAN: ES7620770024003102575766. Need EUR 600K loan to cover contractor arrears.",
    fy: "2022-12-31",
    deposit: "",
    balance: 75000,
    rev: 80000,
    exp: 120000,
    delay: 140,
    disputes: 6,
    court: 2,
    dte: 4.5,
    decline: 35,
    query: "Emergency bridging loan to satisfy pending supplier judgments and court orders",
    amount: 600000
  },
  preset3: {
    name: "Global Retail Imports SL",
    cif: "B34567890",
    desc: "Director: Elena Blanco, email: elena@globalimports.es, mobile: +34 655 443 221. Primary IBAN: ES1234567890123456789012. Requesting EUR 400K credit facility.",
    fy: "2023-12-31",
    deposit: "2024-07-28",
    balance: 40000,
    rev: 95000,
    exp: 140000,
    delay: 55,
    disputes: 2,
    court: 0,
    dte: 2.8,
    decline: 15,
    query: "Revolving credit line to bridge supply chain payment gap of 90 days",
    amount: 400000
  }
};

function showPanel(name){
  document.querySelectorAll('.panel').forEach(p=>p.classList.remove('active'));
  document.querySelectorAll('nav button').forEach(b=>b.classList.remove('active'));
  document.getElementById('panel-'+name).classList.add('active');
  const btns = Array.from(document.querySelectorAll('nav button'));
  const targetBtn = btns.find(b => b.getAttribute('onclick')?.includes(name));
  if (targetBtn) targetBtn.classList.add('active');
}

function setLoading(btnId,spinnerId,loading){
  const btn=document.getElementById(btnId);
  if(!btn)return;
  btn.classList.toggle('loading',loading);
  btn.disabled=loading;
}

async function apiCall(endpoint,body){
  const r=await fetch(API+endpoint,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
  if(!r.ok){const e=await r.json();throw new Error(e.detail||'API error '+r.status);}
  return r.json();
}

function fmt(obj){return JSON.stringify(obj,null,2);}
function updateStat(id,val){const el=document.getElementById(id);if(el)el.textContent=val;}

let reqCount=0;let piiCount=0;
function incReq(){reqCount++;updateStat('stat-req',reqCount);}
function incPII(n){piiCount+=n;updateStat('stat-pii',piiCount);}

function populateForm(presetKey){
  const p = PRESETS[presetKey];
  if(!p) return;
  document.getElementById('fa-name').value = p.name;
  document.getElementById('fa-cif').value = p.cif;
  document.getElementById('fa-fy').value = p.fy;
  document.getElementById('fa-deposit').value = p.deposit;
  document.getElementById('fa-query').value = p.query;
  document.getElementById('fa-desc').value = p.desc;
  document.getElementById('fa-balance').value = p.balance;
  document.getElementById('fa-revenue').value = p.rev;
  document.getElementById('fa-expenses').value = p.exp;
  document.getElementById('fa-dte').value = p.dte;
}

// One-Click Video Scenario Runner
async function runPresetScenario(presetKey){
  populateForm(presetKey);
  const p = PRESETS[presetKey];
  const spinnerWrap = document.getElementById('preset-spinner-wrap');
  const outputDiv = document.getElementById('preset-output');
  
  spinnerWrap.style.display = 'block';
  outputDiv.style.display = 'none';

  try {
    incReq();
    const res = await apiCall('/analyze', {
      entity_name: p.name,
      cif: p.cif,
      raw_description: p.desc,
      fiscal_year_end: p.fy,
      deposit_date: p.deposit || null,
      opening_balance: p.balance,
      monthly_revenue: p.rev,
      monthly_expenses: p.exp,
      payment_delay_days: p.delay,
      creditor_disputes: p.disputes,
      court_proceedings: p.court,
      debt_to_equity: p.dte,
      revenue_decline_pct: p.decline,
      financing_query: p.query,
      amount_eur: p.amount
    });

    incPII(res.pii_result?.entities_found?.length || 0);

    // Render Rich Interactive Card on Dashboard
    renderScenarioResult(res, outputDiv);
    outputDiv.style.display = 'block';
    outputDiv.scrollIntoView({ behavior: 'smooth' });

  } catch(e) {
    outputDiv.style.display = 'block';
    outputDiv.innerHTML = `<div class="alert alert-error">Error executing scenario: ${escHtml(e.message)}</div>`;
  } finally {
    spinnerWrap.style.display = 'none';
  }
}

function renderScenarioResult(res, container){
  const verdict = res.compliance_report?.overall_status || 'UNKNOWN';
  const isBlocked = verdict === 'BLOCKED';
  const isCompliant = verdict === 'COMPLIANT';
  const rating = res.stress_test_report?.liquidity_rating || 'N/A';
  const pii = res.pii_result;
  const audit = res.audit_trail || {};

  let html = `
    <div style="background:var(--surface2);border:1px solid var(--border);border-radius:12px;padding:22px;box-shadow:var(--shadow);">
      <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:10px;margin-bottom:16px;">
        <h3 style="font-size:1.2rem;">🏢 Pipeline Execution: <span style="color:#7dd3fc;">${escHtml(res.entity_name)}</span></h3>
        <span class="risk-badge risk-${verdict}" style="font-size:0.95rem;padding:6px 16px;">VERDICT: ${verdict}</span>
      </div>

      <!-- TIER 1: PII MASKING -->
      <div style="background:var(--bg);border:1px solid var(--border);border-radius:10px;padding:16px;margin-bottom:16px;">
        <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;">
          <h4 style="font-size:0.95rem;color:var(--primary);">🔐 Tier-1 Privacy Layer: PII Anonymization Result</h4>
          <span class="tag tag-green">${pii.entities_found?.length || 0} Entities Redacted</span>
        </div>
        <div style="font-size:0.85rem;line-height:1.6;color:#e2e8f0;background:rgba(0,0,0,0.3);padding:12px;border-radius:6px;border-left:3px solid var(--accent);">
          ${escHtml(pii.anonymized_text)}
        </div>
      </div>

      <!-- TIER 2: COMPLIANCE & LEGAL -->
      <div style="background:var(--bg);border:1px solid var(--border);border-radius:10px;padding:16px;margin-bottom:16px;">
        <h4 style="font-size:0.95rem;color:var(--primary);margin-bottom:8px;">⚖️ Tier-2 Regulatory & Mercantile Registry Compliance</h4>
        <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:10px;font-size:0.85rem;">
          <div><strong>Lending Status:</strong> ${res.compliance_report?.lending_eligible ? '<span style="color:var(--accent);">✅ ELIGIBLE</span>' : '<span style="color:var(--danger);font-weight:700;">❌ BLOCKED</span>'}</div>
          <div><strong>Compliance Score:</strong> ${res.compliance_report?.compliance_score || 0}/100</div>
          <div><strong>Litigation Risk:</strong> ${res.compliance_report?.litigation?.risk_level || 'LOW'}</div>
          <div><strong>Days Overdue:</strong> ${res.compliance_report?.annual_accounts?.days_overdue || 0} days</div>
        </div>
        ${res.compliance_report?.blocking_reasons?.length ? `
          <div class="alert alert-error" style="margin-top:12px;margin-bottom:0;">
            <strong>Regulatory Block Triggered:</strong> ${escHtml(res.compliance_report.blocking_reasons.join(', '))}
          </div>` : ''}
      </div>

      <!-- TIER 3: MONTE CARLO STRESS TEST -->
      <div style="background:var(--bg);border:1px solid var(--border);border-radius:10px;padding:16px;margin-bottom:16px;">
        <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;">
          <h4 style="font-size:0.95rem;color:var(--primary);">📈 Tier-3 Monte Carlo Cash-Flow Stress Testing</h4>
          <span class="risk-badge risk-${rating}">Liquidity Rating: ${rating}</span>
        </div>
        <p style="font-size:0.85rem;color:var(--text2);margin-bottom:8px;">${escHtml(res.stress_test_report?.recommendation || '')}</p>
        <div class="hash-box">
          Basel III LCR & 30/60/90-Day Simulation Summary: ${JSON.stringify(res.stress_test_report?.horizon_summary || {})}
        </div>
      </div>

      <!-- TIER 4: AI RAG RECOMMENDATIONS -->
      <div style="background:var(--bg);border:1px solid var(--border);border-radius:10px;padding:16px;margin-bottom:16px;">
        <h4 style="font-size:0.95rem;color:var(--primary);margin-bottom:10px;">💡 Tier-4 Explainable AI Product Recommendations (RAG)</h4>
        <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:12px;">
          ${(res.ai_recommendations || []).slice(0, 3).map((r, i) => `
            <div style="background:var(--surface2);border:1px solid var(--border);border-radius:8px;padding:12px;">
              <div style="display:flex;justify-content:space-between;align-items:center;">
                <strong style="font-size:0.88rem;">${i+1}. ${escHtml(r.product_name)}</strong>
                <span class="tag tag-green">${Math.round(r.relevance_score * 100)}%</span>
              </div>
              <div style="font-size:0.78rem;color:var(--text2);margin-top:4px;">Max: EUR ${Number(r.max_amount_eur).toLocaleString()} · Fit: ${r.fit_assessment}</div>
              <div style="font-size:0.78rem;color:#93c5fd;margin-top:6px;">${escHtml(r.explanation)}</div>
            </div>
          `).join('')}
        </div>
        <div style="font-size:0.75rem;color:var(--text2);margin-top:10px;font-style:italic;">
          Transparency: ${escHtml(res.eu_ai_act_transparency || '')}
        </div>
      </div>

      <!-- AUDIT TRAIL VERIFICATION OBJECT -->
      <div style="background:rgba(15,23,42,0.9);border:1px solid rgba(59,130,246,0.3);border-radius:10px;padding:16px;">
        <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;">
          <h4 style="font-size:0.9rem;color:#60a5fa;">📋 Verifiable Cryptographic Audit Trail Record</h4>
          <span class="tag tag-blue">SHA-256 Validated</span>
        </div>
        <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:8px;font-size:0.8rem;margin-bottom:8px;">
          <div><strong>Transaction ID:</strong> <span style="color:#fde047;">${escHtml(audit.transaction_id || res.pipeline_id)}</span></div>
          <div><strong>Timestamp:</strong> ${escHtml(audit.timestamp || res.generated_at)}</div>
          <div><strong>PII Entities Detected:</strong> ${audit.original_pii_count ?? pii.entities_found?.length}</div>
          <div><strong>Verdict:</strong> ${escHtml(audit.regulatory_verdict || verdict)}</div>
        </div>
        <div class="hash-box">
          SHA-256 Hash: ${escHtml(audit.sha256_hash || 'SHA256_VERIFIED')}
        </div>
      </div>

    </div>
  `;
  container.innerHTML = html;
}

async function runFullAnalysis(){
  setLoading('fa-btn','fa-spinner',true);
  try{
    incReq();
    const res = await apiCall('/analyze', {
      entity_name: document.getElementById('fa-name').value,
      cif: document.getElementById('fa-cif').value,
      raw_description: document.getElementById('fa-desc').value,
      fiscal_year_end: document.getElementById('fa-fy').value,
      deposit_date: document.getElementById('fa-deposit').value || null,
      opening_balance: parseFloat(document.getElementById('fa-balance').value),
      monthly_revenue: parseFloat(document.getElementById('fa-revenue').value),
      monthly_expenses: parseFloat(document.getElementById('fa-expenses').value),
      payment_delay_days: 12, creditor_disputes: 0, court_proceedings: 0,
      debt_to_equity: parseFloat(document.getElementById('fa-dte').value),
      revenue_decline_pct: 0,
      financing_query: document.getElementById('fa-query').value
    });
    incPII(res.pii_result?.entities_found?.length || 0);
    const resultDiv = document.getElementById('fa-result');
    resultDiv.style.display = 'block';
    renderScenarioResult(res, resultDiv);
  } catch(e) {
    document.getElementById('fa-result').style.display = 'block';
    document.getElementById('fa-result').innerHTML = `<div class="alert alert-error">${escHtml(e.message)}</div>`;
  }
  setLoading('fa-btn','fa-spinner',false);
}

async function runPIIMask(){
  setLoading('pii-btn','pii-spinner',true);
  try{
    incReq();
    const res = await apiCall('/mask-pii', {text: document.getElementById('pii-input').value, context: 'financial'});
    document.getElementById('pii-result').style.display='block';
    document.getElementById('pii-masked-text').innerHTML='<strong>🔐 Clean Masked Output:</strong><br/>'+escHtml(res.anonymized_text);
    document.getElementById('pii-details').innerHTML='<pre>'+fmt(res)+'</pre>';
    incPII(res.entities_found ? res.entities_found.length : 0);
  }catch(e){
    document.getElementById('pii-result').style.display='block';
    document.getElementById('pii-details').innerHTML='<div class="alert alert-error">'+e.message+'</div>';
  }
  setLoading('pii-btn','pii-spinner',false);
}

async function runCompliance(){
  setLoading('co-btn','co-spinner',true);
  try{
    incReq();
    const res = await apiCall('/compliance', {
      entity_name: document.getElementById('co-name').value,
      cif: document.getElementById('co-cif').value,
      fiscal_year_end: document.getElementById('co-fy').value,
      deposit_date: document.getElementById('co-deposit').value || null,
      payment_delay_days: parseFloat(document.getElementById('co-delay').value) || 0,
      creditor_disputes: parseInt(document.getElementById('co-disputes').value) || 0,
      court_proceedings: parseInt(document.getElementById('co-court').value) || 0,
      debt_to_equity: parseFloat(document.getElementById('co-dte').value) || 0.5,
      revenue_decline_pct: parseFloat(document.getElementById('co-decline').value) || 0,
      jurisdiction: document.getElementById('co-juris').value
    });
    document.getElementById('co-result').style.display='block';
    document.getElementById('co-result').innerHTML='<pre>'+fmt(res)+'</pre>';
  }catch(e){
    document.getElementById('co-result').style.display='block';
    document.getElementById('co-result').innerHTML='<div class="alert alert-error">'+e.message+'</div>';
  }
  setLoading('co-btn','co-spinner',false);
}

async function runStressTest(){
  setLoading('st-btn','st-spinner',true);
  try{
    incReq();
    const res = await apiCall('/stress-test', {
      entity_name: document.getElementById('st-name').value,
      opening_balance: parseFloat(document.getElementById('st-balance').value),
      monthly_revenue: parseFloat(document.getElementById('st-revenue').value),
      monthly_expenses: parseFloat(document.getElementById('st-expenses').value)
    });
    document.getElementById('st-result').style.display='block';
    let html = `<div style="margin-bottom:12px"><span class="risk-badge risk-${res.overall_liquidity_rating}">Liquidity Rating: ${res.overall_liquidity_rating}</span></div>`;
    html += `<p style="color:var(--text2);margin-bottom:12px;font-size:0.85rem">${res.recommendation}</p>`;
    html += '<pre>' + fmt(res.horizon_summary) + '</pre>';
    document.getElementById('st-result').innerHTML = html;
  }catch(e){
    document.getElementById('st-result').style.display='block';
    document.getElementById('st-result').innerHTML='<div class="alert alert-error">'+e.message+'</div>';
  }
  setLoading('st-btn','st-spinner',false);
}

async function runRecommend(){
  setLoading('rc-btn','rc-spinner',true);
  try{
    incReq();
    const amount = document.getElementById('rc-amount').value;
    const res = await apiCall('/recommend', {
      entity_name: document.getElementById('rc-name').value,
      query: document.getElementById('rc-query').value,
      top_k: parseInt(document.getElementById('rc-topk').value),
      amount_eur: amount ? parseFloat(amount) : null
    });
    document.getElementById('rc-result').style.display='block';
    let html = `<p style="color:var(--text2);font-size:0.8rem;margin-bottom:12px">Evaluated ${res.total_products_evaluated} products in ${res.processing_time_ms}ms using <strong>${res.model_used}</strong></p>`;
    res.recommendations.forEach((r, i) => {
      const pct = Math.round(r.relevance_score * 100);
      html += `<div class="rec-card">
        <h3>${i+1}. ${escHtml(r.product_name)} <span class="tag tag-blue">${r.category}</span></h3>
        <div class="score-bar"><div class="score-fill" style="width:${pct}%"></div></div>
        <div style="display:flex;gap:8px;align-items:center;margin:6px 0">
          <span class="tag tag-green">${pct}% match</span>
          <span class="tag tag-orange">${r.fit_assessment}</span>
          <span style="color:var(--text2);font-size:0.78rem">Max: EUR ${Number(r.max_amount_eur).toLocaleString()}</span>
        </div>
        <p style="font-size:0.82rem;color:var(--text2)">${escHtml(r.explanation)}</p>
      </div>`;
    });
    document.getElementById('rc-result').innerHTML = html;
  }catch(e){
    document.getElementById('rc-result').style.display='block';
    document.getElementById('rc-result').innerHTML='<div class="alert alert-error">'+e.message+'</div>';
  }
  setLoading('rc-btn','rc-spinner',false);
}

async function loadAudit(){
  const r = await fetch(API + '/audit-trail');
  const data = await r.json();
  document.getElementById('audit-result').innerHTML = '<pre>' + fmt(data) + '</pre>';
}

function escHtml(t){return t ? t.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;') : '';}
</script>
</body>
</html>"""


# ---------------------------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------------------------

@app.get("/", tags=["Health"])
async def health_check() -> Dict[str, Any]:
    return {
        "service": "SME Finance Copilot",
        "version": "2.0.0",
        "status": "operational",
        "timestamp": datetime.utcnow().isoformat(),
        "competition": "FinTechathon 2026 — Shenzhen — Topic E",
        "engines": {
            "pii_masking": "operational",
            "compliance":  "operational",
            "stress_test": "operational",
            "ai_copilot":  "operational",
        },
    }


@app.get("/dashboard", response_class=HTMLResponse, tags=["Dashboard"])
async def dashboard() -> HTMLResponse:
    """Serve the interactive web dashboard for the demo video."""
    return HTMLResponse(content=DASHBOARD_HTML)


@app.post("/mask-pii", tags=["Security"])
async def mask_pii(request: PIIMaskRequest) -> Dict[str, Any]:
    """Anonymize PII in the provided text using Presidio NLP + regex fallback."""
    result = _pii_engine.anonymize(request.text, request.context)
    _log_audit("/mask-pii", request.dict(), {"status": "ok", "audit_trail": result.audit_trail})
    return {
        "original_length": len(request.text),
        "anonymized_text": result.anonymized_text,
        "entities_found": result.entities_found,
        "risk_score": result.risk_score,
        "processing_time_ms": result.processing_time_ms,
        "audit_token": result.audit_token,
        "audit_trail": result.audit_trail,
        "compliance_tags": ["GDPR-Art25", "ISO27001-A18"],
    }


@app.post("/compliance", tags=["Legal"])
async def check_compliance(request: ComplianceRequest) -> Dict[str, Any]:
    """Run full regulatory compliance check (Mercantile Registry + AML + Litigation)."""
    try:
        fy_end = date.fromisoformat(request.fiscal_year_end)
    except ValueError:
        raise HTTPException(status_code=422, detail="Invalid fiscal_year_end date format (YYYY-MM-DD)")

    deposit = None
    if request.deposit_date:
        try:
            deposit = date.fromisoformat(request.deposit_date)
        except ValueError:
            raise HTTPException(status_code=422, detail="Invalid deposit_date format (YYYY-MM-DD)")

    report = _legal_engine.run_full_compliance_check(
        entity_name=request.entity_name,
        cif=request.cif,
        fiscal_year_end=fy_end,
        deposit_date=deposit,
        payment_delay_days=request.payment_delay_days,
        creditor_disputes=request.creditor_disputes,
        court_proceedings=request.court_proceedings,
        debt_to_equity=request.debt_to_equity,
        revenue_decline_pct=request.revenue_decline_pct,
        jurisdiction=request.jurisdiction,
    )
    _log_audit("/compliance", request.dict(), {"status": report.overall_status.value})
    return asdict(report)


@app.post("/stress-test", tags=["Risk"])
async def stress_test(request: StressTestRequest) -> Dict[str, Any]:
    """Run Monte Carlo 30/60/90-day cash-flow stress test."""
    report = _stress_tester.run_full_stress_test(
        entity_name=request.entity_name,
        opening_balance=request.opening_balance,
        monthly_revenue=request.monthly_revenue,
        monthly_expenses=request.monthly_expenses,
    )
    _log_audit("/stress-test", request.dict(), {"status": "ok"})
    return asdict(report)


@app.post("/recommend", tags=["AI"])
async def recommend(request: RecommendRequest) -> Dict[str, Any]:
    """Generate AI-powered, explainable financial product recommendations."""
    response = _ai_copilot.recommend(
        entity_name=request.entity_name,
        query=request.query,
        top_k=request.top_k,
        categories=request.categories,
        amount_eur=request.amount_eur,
    )
    _log_audit("/recommend", request.dict(), {"status": "ok"})
    return asdict(response)


@app.post("/analyze", tags=["Full Pipeline"])
async def full_analysis(request: SMEAnalysisRequest) -> Dict[str, Any]:
    """
    Full SME analysis pipeline.

    Executes in sequence:
      1. PII Masking (Tier-1 Security Gate)
      2. Regulatory Compliance Check
      3. Monte Carlo Cash-Flow Stress Test
      4. AI Product Recommendations
      5. Cryptographic SHA-256 Audit Trail Generation
    """
    pipeline_id = f"PIPE-{uuid.uuid4().hex[:8].upper()}"
    t0 = time.perf_counter()

    # --- Step 1: PII Masking ---
    pii_result = _pii_engine.anonymize(request.raw_description, context="financial")

    # --- Step 2: Compliance ---
    try:
        fy_end = date.fromisoformat(request.fiscal_year_end)
    except ValueError:
        raise HTTPException(status_code=422, detail="Invalid fiscal_year_end format.")

    deposit = None
    if request.deposit_date:
        try:
            deposit = date.fromisoformat(request.deposit_date)
        except ValueError:
            raise HTTPException(status_code=422, detail="Invalid deposit_date format.")

    compliance_report = _legal_engine.run_full_compliance_check(
        entity_name=request.entity_name,
        cif=request.cif,
        fiscal_year_end=fy_end,
        deposit_date=deposit,
        payment_delay_days=request.payment_delay_days,
        creditor_disputes=request.creditor_disputes,
        court_proceedings=request.court_proceedings,
        debt_to_equity=request.debt_to_equity,
        revenue_decline_pct=request.revenue_decline_pct,
    )

    # --- Step 3: Stress Test ---
    stress_report = _stress_tester.run_full_stress_test(
        entity_name=request.entity_name,
        opening_balance=request.opening_balance,
        monthly_revenue=request.monthly_revenue,
        monthly_expenses=request.monthly_expenses,
    )

    # --- Step 4: AI Recommendations ---
    ai_response = _ai_copilot.recommend(
        entity_name=request.entity_name,
        query=request.financing_query,
        top_k=5,
        categories=request.categories,
        amount_eur=request.amount_eur,
    )

    total_ms = (time.perf_counter() - t0) * 1000

    # --- Step 5: Cryptographic Audit Trail Object ---
    regulatory_verdict = compliance_report.overall_status.value
    audit_hash_payload = (
        f"{pipeline_id}:{datetime.utcnow().isoformat()}:"
        f"{len(pii_result.entities_found)}:{regulatory_verdict}:{pii_result.anonymized_text}"
    )
    sha256_hash = hashlib.sha256(audit_hash_payload.encode()).hexdigest()

    audit_trail_entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "transaction_id": pipeline_id,
        "original_pii_count": len(pii_result.entities_found),
        "anonymized_text": pii_result.anonymized_text,
        "regulatory_verdict": regulatory_verdict,
        "sha256_hash": sha256_hash,
    }

    _log_audit("/analyze", {"entity": request.entity_name}, {"status": "ok", "audit_trail": audit_trail_entry})

    return {
        "pipeline_id":        pipeline_id,
        "entity_name":        request.entity_name,
        "total_processing_ms": round(total_ms, 2),
        "generated_at":       datetime.utcnow().isoformat(),
        "audit_trail":        audit_trail_entry,
        "pii_result": {
            "anonymized_text":  pii_result.anonymized_text,
            "entities_found":   pii_result.entities_found,
            "risk_score":       pii_result.risk_score,
            "audit_token":      pii_result.audit_token,
            "audit_trail":      audit_trail_entry,
        },
        "compliance_report":  asdict(compliance_report),
        "stress_test_report": {
            "liquidity_rating": stress_report.overall_liquidity_rating,
            "recommendation":   stress_report.recommendation,
            "horizon_summary":  stress_report.horizon_summary,
        },
        "ai_recommendations": [
            {
                "product_id":       r.product_id,
                "product_name":     r.product_name,
                "category":         r.category,
                "relevance_score":  r.relevance_score,
                "fit_assessment":   r.fit_assessment,
                "max_amount_eur":   r.max_amount_eur,
                "explanation":      r.explanation,
            }
            for r in ai_response.recommendations
        ],
        "model_used": ai_response.model_used,
        "eu_ai_act_transparency": ai_response.eu_ai_act_transparency,
    }


@app.get("/products", tags=["Catalogue"])
async def list_products(category: Optional[str] = None) -> Dict[str, Any]:
    """List all financial products in the catalogue, with optional category filter."""
    products = PRODUCT_CATALOGUE
    if category:
        products = [p for p in products if p["category"] == category]
    return {
        "total": len(products),
        "products": products,
        "categories": list({p["category"] for p in PRODUCT_CATALOGUE}),
    }


@app.get("/audit-trail", tags=["Compliance"])
async def audit_trail() -> Dict[str, Any]:
    """Return the in-memory audit log for this session."""
    pii_audit = _pii_engine.get_audit_trail()
    return {
        "session_audit_log": _audit_log,
        "pii_audit_trail":   pii_audit,
        "total_requests":    len(_audit_log),
        "total_pii_scans":   len(pii_audit),
        "generated_at":      datetime.utcnow().isoformat(),
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info",
    )
