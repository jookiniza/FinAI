# SME Finance Copilot

**FinTechathon 2026 — Shenzhen International FinTech Competition | International Track — Topic E**

> An enterprise-grade, AI-powered financial copilot for Small & Medium Enterprises (SMEs). Maximizes access to capital through intelligent product matching, proactive regulatory compliance, and Monte Carlo risk modeling — all within a GDPR-compliant, privacy-first architecture.

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    SME Finance Copilot — v2.0.0                          │
│              FastAPI Orchestrator + Interactive Web Dashboard             │
└────────────────────────────┬────────────────────────────────────────────┘
                             │  POST /analyze  (Full Pipeline)
             ┌───────────────▼───────────────────────────────┐
             │          TIER-1: Security Gate                  │
             │   PrivacyPreservingLayer  (security_layer.py)  │
             │   ┌──────────────────────────────────────────┐ │
             │   │  Microsoft Presidio NLP Engine            │ │
             │   │  + spaCy en_core_web_lg model             │ │
             │   │  + Regex fallback (15 PII pattern types)  │ │
             │   │  GDPR Art.25 | ISO 27001 A.18             │ │
             │   └──────────────────────────────────────────┘ │
             └──────────────┬───────────────────────┬─────────┘
                            │                       │
         ┌──────────────────▼──────┐   ┌────────────▼────────────────┐
         │  Compliance Engine       │   │  AI RAG Copilot              │
         │  (legal_engine.py)       │   │  (ai_agent.py)               │
         │  ┌─────────────────────┐ │   │  ┌──────────────────────────┐│
         │  │ MercantileRegistry  │ │   │  │ EmbeddingEngine          ││
         │  │ LitigationScorer    │ │   │  │ all-MiniLM-L6-v2         ││
         │  │ AMLScreener         │ │   │  │ (TF-IDF fallback)        ││
         │  │ 6AMLD | LSC Art.279 │ │   │  ├──────────────────────────┤│
         │  └─────────────────────┘ │   │  │ LendingAgent             ││
         └──────────────────────────┘   │  │ GrantAgent               ││
                                        │  │ InsuranceAgent           ││
                                        │  │ EU AI Act Art.13         ││
                                        │  └──────────────────────────┘│
                                        └────────────────────────────────┘
                                                      │
                            ┌─────────────────────────▼──────────────────┐
                            │      Monte Carlo Stress Testing Engine       │
                            │      (cashflow_engine.py)                    │
                            │  ┌──────────────────────────────────────┐   │
                            │  │  MonteCarloCashFlow (5,000 sims)     │   │
                            │  │  3 Scenarios × 3 Horizons = 9 runs   │   │
                            │  │  VaR95, VaR99, CVaR, P(ruin), LCR   │   │
                            │  │  Basel III LCR | EBA ILAAP           │   │
                            │  └──────────────────────────────────────┘   │
                            └────────────────────────────────────────────┘
```

---

## Key Features

| Feature | Detail |
|---|---|
| 🔐 **PII Masking** | Microsoft Presidio + spaCy NLP; 15+ entity types; GDPR Art.25 compliant |
| ⚖️ **Regulatory Compliance** | Spanish Mercantile Registry (LSC Art.279), AML 6AMLD, Litigation Risk |
| 📊 **Monte Carlo Stress Test** | 5,000 simulations × 3 scenarios × 30/60/90-day horizons |
| 💡 **AI Recommendations** | RAG with `all-MiniLM-L6-v2`; 10-product catalogue; explainable scores |
| 🌐 **Interactive Dashboard** | Self-contained HTML/JS at `/dashboard`; live API calls |
| 📋 **Audit Trail** | Immutable per-request audit tokens; session-level compliance export |
| 🤖 **EU AI Act Compliant** | Art.13 transparency disclosure on every AI-generated response |

---

## Project Structure

```
sme-finance-agent/
├── main.py                      # FastAPI orchestrator + web dashboard
├── security_layer.py            # PII masking engine (Presidio)
├── legal_engine.py              # Regulatory compliance engine
├── cashflow_engine.py           # Monte Carlo stress testing
├── ai_agent.py                  # Multi-agent RAG copilot
├── test_main.py                 # Pytest test suite
├── requirements.txt             # Pinned dependencies
├── execution_evidence.log       # Auto-generated test evidence (after pytest)
├── README.md                    # This file
├── Security_Self_Assessment_Report.md
├── presentation_deck_script.md
└── demo_video_script.md
```

---

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Download spaCy Language Model

```bash
python -m spacy download en_core_web_lg
```

### 3. Run the API Server

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### 4. Open the Dashboard

Navigate to **[http://localhost:8000/dashboard](http://localhost:8000/dashboard)**

### 5. Run Tests & Generate Evidence Log

```bash
pytest test_main.py -v
```

Evidence is saved to `execution_evidence.log`.

---

## API Reference

### Base URL
`http://localhost:8000`

### Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/` | Health check + engine status |
| `GET` | `/dashboard` | Interactive HTML/JS dashboard |
| `POST` | `/analyze` | **Full 4-engine pipeline** |
| `POST` | `/mask-pii` | PII anonymization only |
| `POST` | `/compliance` | Regulatory compliance check |
| `POST` | `/stress-test` | Monte Carlo stress test |
| `POST` | `/recommend` | AI product recommendations |
| `GET` | `/products` | Product catalogue |
| `GET` | `/audit-trail` | Session audit log |
| `GET` | `/docs` | Swagger UI (auto-generated) |
| `GET` | `/redoc` | ReDoc API documentation |

### Example: Full Pipeline Request

```bash
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "entity_name": "Acme Logistics SL",
    "cif": "B12345678",
    "raw_description": "CEO John Smith at ceo@acme.com, +34 612 345 678. IBAN ES9121000418401234567891",
    "fiscal_year_end": "2023-12-31",
    "opening_balance": 250000,
    "monthly_revenue": 180000,
    "monthly_expenses": 155000,
    "payment_delay_days": 45,
    "creditor_disputes": 1,
    "court_proceedings": 0,
    "debt_to_equity": 1.8,
    "revenue_decline_pct": 5.0,
    "financing_query": "We need working capital for seasonal cash flow gaps"
  }'
```

### Example: PII Masking

```bash
curl -X POST http://localhost:8000/mask-pii \
  -H "Content-Type: application/json" \
  -d '{"text": "Call CEO John Smith at +34 612 345 678 or john@company.com"}'
```

**Response:**
```json
{
  "anonymized_text": "Call CEO [PERSON_REDACTED] at [PHONE_REDACTED] or [EMAIL_REDACTED]",
  "entities_found": [...],
  "risk_score": 0.65,
  "audit_token": "a3f8b2c1d4e5",
  "compliance_tags": ["GDPR-Art25", "ISO27001-A18"]
}
```

---

## Compliance & Security

| Standard | Coverage |
|---|---|
| GDPR Art. 25 | Data Protection by Design — PII masked before processing |
| ISO 27001 A.18 | Compliance with legal requirements |
| PCI DSS v4 | Card data never stored; immediately anonymized |
| 6AMLD | AML watchlist screening on every entity |
| EU AI Act Art. 13 | Transparency disclosure on all AI outputs |
| Basel III LCR | Liquidity Coverage Ratio proxy in stress tests |
| EBA ILAAP | Internal Liquidity Adequacy Assessment methodology |
| LSC Art. 279 | Spanish annual accounts deposit deadline enforcement |

---

## Scoring Alignment

| Competition Category | Score Weight | Our Implementation |
|---|---|---|
| Task Completion | 40% | Full 4-engine pipeline; 10-product RAG; real-time API |
| Innovation & Interaction | 30% | Live dashboard; Monte Carlo; explainable AI; multi-agent |
| Security & Compliance | 30% | Tier-1 PII gate; AML; litigation risk; 8 regulatory standards |

---

## Technology Stack

- **Runtime**: Python 3.11+
- **API Framework**: FastAPI 0.111
- **NLP / AI**: sentence-transformers (`all-MiniLM-L6-v2`), spaCy (`en_core_web_lg`)
- **PII Engine**: Microsoft Presidio Analyzer + Anonymizer
- **Risk Modeling**: NumPy (Monte Carlo), scikit-learn (TF-IDF fallback)
- **Testing**: pytest 8.2, FastAPI TestClient
- **Frontend**: Vanilla HTML5/CSS3/JS (zero framework dependency)

---

## License

MIT License — FinTechathon 2026 Submission

---

*Built for the 2026 Shenzhen International FinTech Competition — International Track — Topic E: SME Finance Copilot*
