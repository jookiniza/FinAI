"""
ai_agent.py
===========
Multi-Agent Financial Matching Core — Explainable RAG Product Recommendations.

Architecture:
  - SentenceTransformer embeddings for semantic product-query matching
  - Cosine-similarity ranking with explainability tokens
  - Retrieval-Augmented Generation (RAG) context assembly
  - Three specialist sub-agents: LendingAgent, GrantAgent, InsuranceAgent
  - Orchestrator: SMEFinancialCopilot — routes queries and merges results

Models:
  - Primary  : all-MiniLM-L6-v2  (fast, 80MB, 384-dim)
  - Fallback : TF-IDF + cosine similarity (no ML dependencies)

Compliance: Results are explainable per EU AI Act Art. 13 (transparency).
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional sentence-transformers
# ---------------------------------------------------------------------------
try:
    from sentence_transformers import SentenceTransformer, util as st_util
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logger.warning(
        "sentence-transformers not installed. Using TF-IDF cosine fallback."
    )

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


# ---------------------------------------------------------------------------
# Financial Product Catalogue (SME-focused)
# ---------------------------------------------------------------------------

PRODUCT_CATALOGUE: List[Dict[str, Any]] = [
    # ---- Lending products ----
    {
        "id": "LOAN-ICO-01",
        "category": "lending",
        "name": "ICO Working Capital Loan",
        "description": (
            "Spanish Institute of Credit (ICO) backed SME working capital loan. "
            "Covers operational expenses, payroll, and short-term liquidity gaps. "
            "Up to 1.5M EUR, 5-year term, subsidised interest rate 2.5-4.5%."
        ),
        "max_amount_eur": 1_500_000,
        "term_months": 60,
        "interest_rate_pct": 3.5,
        "requirements": ["2 years trading", "positive EBITDA", "no active insolvency"],
        "tags": ["liquidity", "working-capital", "government-backed", "ICO"],
    },
    {
        "id": "LOAN-BEI-02",
        "category": "lending",
        "name": "EIB SME Innovation Loan",
        "description": (
            "European Investment Bank loan for innovative SMEs investing in R&D, "
            "digitalisation, or green transition. Up to 12.5M EUR, 12-year term. "
            "Reduced collateral requirements for certified innovative enterprises."
        ),
        "max_amount_eur": 12_500_000,
        "term_months": 144,
        "interest_rate_pct": 2.1,
        "requirements": ["R&D investment > 5% revenue", "EU-registered", "< 250 employees"],
        "tags": ["innovation", "R&D", "digitalisation", "green", "EIB", "long-term"],
    },
    {
        "id": "LOAN-REV-03",
        "category": "lending",
        "name": "Revolving Credit Facility",
        "description": (
            "Flexible revolving credit line for SMEs with seasonal cash-flow patterns. "
            "Draw-down on demand up to 500K EUR. Interest charged only on drawn balance. "
            "Ideal for import/export businesses with 60-90 day payment cycles."
        ),
        "max_amount_eur": 500_000,
        "term_months": 12,
        "interest_rate_pct": 5.2,
        "requirements": ["18 months trading", "annual revenue > 300K"],
        "tags": ["revolving", "seasonal", "flexible", "import", "export", "working-capital"],
    },
    {
        "id": "LOAN-GRN-04",
        "category": "lending",
        "name": "Green Transition SME Loan",
        "description": (
            "Subsidised loan for SMEs investing in renewable energy, EV fleet, "
            "or energy efficiency upgrades. Qualifies for EU Taxonomy green label. "
            "Up to 2M EUR, fixed rate 1.8%, 10-year term with 2-year grace period."
        ),
        "max_amount_eur": 2_000_000,
        "term_months": 120,
        "interest_rate_pct": 1.8,
        "requirements": ["documented green capex plan", "EU Taxonomy alignment"],
        "tags": ["green", "sustainability", "ESG", "renewable", "energy-efficiency"],
    },
    # ---- Grants ----
    {
        "id": "GRANT-H2EU-05",
        "category": "grant",
        "name": "Horizon Europe SME Instrument (EIC Accelerator)",
        "description": (
            "EU non-dilutive grant up to 2.5M EUR + optional equity investment "
            "for highly innovative deep-tech SMEs. Open to all EU member states. "
            "Success rate ~5%: requires strong innovation case and market validation."
        ),
        "max_amount_eur": 2_500_000,
        "term_months": 0,
        "interest_rate_pct": 0.0,
        "requirements": ["TRL 5-8", "scalable business model", "full-time founder team"],
        "tags": ["grant", "innovation", "deep-tech", "EU", "non-dilutive", "EIC"],
    },
    {
        "id": "GRANT-CDTI-06",
        "category": "grant",
        "name": "CDTI NEOTEC Grant (Spain)",
        "description": (
            "Spanish CDTI grant for newly established innovative companies (< 3 years). "
            "Up to 250K EUR non-repayable for technology-based startups. "
            "Fast-track review for AI, biotech, and cleantech sectors."
        ),
        "max_amount_eur": 250_000,
        "term_months": 0,
        "interest_rate_pct": 0.0,
        "requirements": ["incorporated < 3 years", "technology focus", "Spain-registered"],
        "tags": ["grant", "startup", "technology", "CDTI", "Spain", "non-dilutive"],
    },
    {
        "id": "GRANT-DGIPYME-07",
        "category": "grant",
        "name": "DGIPYME Digitalisation Voucher",
        "description": (
            "Spanish Ministry of Industry voucher for SME digitalisation. "
            "Covers CRM, ERP, cybersecurity, and e-commerce implementation. "
            "Up to 12K EUR for micro-enterprises, 29K for small businesses."
        ),
        "max_amount_eur": 29_000,
        "term_months": 0,
        "interest_rate_pct": 0.0,
        "requirements": ["registered SME in Spain", "specific digital service categories"],
        "tags": ["digitalisation", "voucher", "CRM", "ERP", "cybersecurity", "Spain"],
    },
    # ---- Insurance ----
    {
        "id": "INS-CREDIT-08",
        "category": "insurance",
        "name": "Trade Credit Insurance (CESCE)",
        "description": (
            "Spanish state-backed export credit insurance covering non-payment risk "
            "from domestic and international buyers. Covers up to 85% of insured risk. "
            "Enables SMEs to offer open-account terms and win larger contracts."
        ),
        "max_amount_eur": 5_000_000,
        "term_months": 12,
        "interest_rate_pct": 0.0,
        "requirements": ["active export activity or domestic B2B sales", "CESCE KYB pass"],
        "tags": ["insurance", "credit-risk", "export", "trade", "CESCE", "non-payment"],
    },
    {
        "id": "INS-SURETY-09",
        "category": "insurance",
        "name": "Performance Bond / Surety Insurance",
        "description": (
            "Guarantee bond for SMEs bidding on public procurement contracts. "
            "Replaces cash deposits required by contracting authorities. "
            "Frees working capital while maintaining regulatory compliance."
        ),
        "max_amount_eur": 10_000_000,
        "term_months": 24,
        "interest_rate_pct": 0.0,
        "requirements": ["public procurement contract award", "3 years trading history"],
        "tags": ["surety", "bond", "procurement", "public-sector", "guarantee"],
    },
    {
        "id": "INS-BIZ-10",
        "category": "insurance",
        "name": "Business Interruption Insurance",
        "description": (
            "Covers revenue loss during operational disruptions caused by fire, "
            "flood, cyber attacks, or supply-chain failures. Pays ongoing fixed costs "
            "and lost profit margin during recovery period (up to 24 months)."
        ),
        "max_amount_eur": 2_000_000,
        "term_months": 12,
        "interest_rate_pct": 0.0,
        "requirements": ["property insurance in place", "business continuity plan"],
        "tags": ["business-interruption", "cyber", "supply-chain", "resilience"],
    },
]


# ---------------------------------------------------------------------------
# Embedding engine (with TF-IDF fallback)
# ---------------------------------------------------------------------------

class EmbeddingEngine:
    """Wraps sentence-transformers with a TF-IDF fallback."""

    MODEL_NAME = "all-MiniLM-L6-v2"

    def __init__(self) -> None:
        self._model: Optional[Any] = None
        self._tfidf: Optional[Any] = None
        self._tfidf_matrix: Optional[Any] = None
        self._corpus: List[str] = []

        if TRANSFORMERS_AVAILABLE:
            try:
                self._model = SentenceTransformer(self.MODEL_NAME)
                logger.info("SentenceTransformer loaded: %s", self.MODEL_NAME)
            except Exception as exc:
                logger.warning("ST load failed: %s", exc)
                self._model = None

        if self._model is None:
            self._init_tfidf()

    def _init_tfidf(self) -> None:
        if SKLEARN_AVAILABLE:
            self._tfidf = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
            logger.info("TF-IDF fallback engine initialised.")
        else:
            logger.warning("Neither sentence-transformers nor scikit-learn available. Using keyword matching.")

    def fit_corpus(self, texts: List[str]) -> None:
        self._corpus = texts
        if self._tfidf:
            self._tfidf_matrix = self._tfidf.fit_transform(texts)

    def encode(self, texts: List[str]) -> Any:
        if self._model:
            return self._model.encode(texts, convert_to_tensor=True, show_progress_bar=False)
        return None

    def compute_similarities(self, query: str, corpus_texts: List[str]) -> List[float]:
        """Return cosine similarity scores between query and each corpus text."""
        if self._model:
            try:
                q_emb = self._model.encode([query], convert_to_tensor=True)
                c_emb = self._model.encode(corpus_texts, convert_to_tensor=True)
                scores = st_util.cos_sim(q_emb, c_emb)[0].tolist()
                return scores
            except Exception as exc:
                logger.warning("ST inference failed: %s", exc)

        if self._tfidf and self._tfidf_matrix is not None:
            try:
                q_vec = self._tfidf.transform([query])
                from sklearn.metrics.pairwise import cosine_similarity as cossim
                sims = cossim(q_vec, self._tfidf_matrix)[0].tolist()
                return sims
            except Exception:
                pass

        # Final fallback: keyword overlap
        query_words = set(query.lower().split())
        scores = []
        for text in corpus_texts:
            text_words = set(text.lower().split())
            overlap = len(query_words & text_words)
            scores.append(overlap / max(len(query_words), 1))
        return scores


# ---------------------------------------------------------------------------
# Data-classes for recommendations
# ---------------------------------------------------------------------------

@dataclass
class ProductRecommendation:
    product_id: str
    product_name: str
    category: str
    relevance_score: float       # 0.0 – 1.0
    explanation: str
    matching_keywords: List[str]
    max_amount_eur: float
    interest_rate_pct: float
    requirements: List[str]
    fit_assessment: str          # "Excellent" / "Good" / "Moderate" / "Low"


@dataclass
class RAGRecommendationResponse:
    query_id: str
    entity_name: str
    query: str
    recommendations: List[ProductRecommendation]
    rag_context: str             # assembled context for LLM (audit trail)
    total_products_evaluated: int
    processing_time_ms: float
    model_used: str
    generated_at: str
    eu_ai_act_transparency: str  # EU AI Act Art. 13 disclosure


# ---------------------------------------------------------------------------
# Specialist Sub-Agents
# ---------------------------------------------------------------------------

class BaseFinancialAgent:
    """Base class for specialist financial sub-agents."""

    category: str = "all"
    CATALOGUE = PRODUCT_CATALOGUE

    def __init__(self, engine: EmbeddingEngine) -> None:
        self._engine = engine
        self._products = [p for p in self.CATALOGUE if p["category"] == self.category or self.category == "all"]
        self._corpus = [self._product_to_text(p) for p in self._products]
        self._engine.fit_corpus(self._corpus)

    @staticmethod
    def _product_to_text(product: Dict[str, Any]) -> str:
        return (
            f"{product['name']}. {product['description']} "
            f"Tags: {', '.join(product.get('tags', []))}. "
            f"Requirements: {', '.join(product.get('requirements', []))}."
        )

    def recommend(self, query: str, top_k: int = 3) -> List[ProductRecommendation]:
        if not self._products:
            return []

        scores = self._engine.compute_similarities(query, self._corpus)
        ranked = sorted(zip(self._products, scores), key=lambda x: x[1], reverse=True)

        results = []
        for product, score in ranked[:top_k]:
            # Build explanation tokens
            query_words = set(query.lower().split())
            product_tags = set(product.get("tags", []))
            matching = list(query_words & product_tags)

            if score >= 0.70:
                fit = "Excellent"
            elif score >= 0.50:
                fit = "Good"
            elif score >= 0.30:
                fit = "Moderate"
            else:
                fit = "Low"

            explanation = (
                f"Matched with {fit.lower()} relevance (score={score:.3f}). "
                f"Key alignment: {', '.join(matching) if matching else 'semantic similarity'}. "
                f"Product covers: {product['description'][:120]}..."
            )

            results.append(ProductRecommendation(
                product_id=product["id"],
                product_name=product["name"],
                category=product["category"],
                relevance_score=round(score, 4),
                explanation=explanation,
                matching_keywords=matching,
                max_amount_eur=product.get("max_amount_eur", 0),
                interest_rate_pct=product.get("interest_rate_pct", 0.0),
                requirements=product.get("requirements", []),
                fit_assessment=fit,
            ))

        return results


class LendingAgent(BaseFinancialAgent):
    """Specialist agent for credit and lending products."""
    category = "lending"


class GrantAgent(BaseFinancialAgent):
    """Specialist agent for grants and subsidies."""
    category = "grant"


class InsuranceAgent(BaseFinancialAgent):
    """Specialist agent for insurance and guarantee products."""
    category = "insurance"


class GeneralistAgent(BaseFinancialAgent):
    """Cross-category agent for broad SME finance queries."""
    category = "all"


# ---------------------------------------------------------------------------
# Main RAG Orchestrator
# ---------------------------------------------------------------------------

class SMEFinancialCopilot:
    """
    Multi-Agent Financial Matching Core.

    Routes incoming SME finance queries to specialist sub-agents,
    assembles RAG context, and returns explainable ranked recommendations.

    Usage
    -----
    >>> copilot = SMEFinancialCopilot()
    >>> response = copilot.recommend(
    ...     entity_name="Acme Logistics SL",
    ...     query="We need working capital to cover 90-day payment cycles",
    ...     top_k=5,
    ... )
    """

    EU_AI_ACT_DISCLOSURE = (
        "AI-generated recommendations. Produced by a RAG system using semantic "
        "similarity matching (all-MiniLM-L6-v2). Not regulated financial advice. "
        "Compliant with EU AI Act Art. 13 transparency requirements."
    )

    def __init__(self) -> None:
        self._engine = EmbeddingEngine()
        self._agents = {
            "lending":   LendingAgent(EmbeddingEngine()),
            "grant":     GrantAgent(EmbeddingEngine()),
            "insurance": InsuranceAgent(EmbeddingEngine()),
            "all":       GeneralistAgent(EmbeddingEngine()),
        }
        model_name = (
            EmbeddingEngine.MODEL_NAME
            if TRANSFORMERS_AVAILABLE
            else ("TF-IDF" if SKLEARN_AVAILABLE else "keyword-overlap")
        )
        self._model_name = model_name
        logger.info("SMEFinancialCopilot initialised. Model: %s", model_name)

    def recommend(
        self,
        entity_name: str,
        query: str,
        top_k: int = 5,
        categories: Optional[List[str]] = None,
        amount_eur: Optional[float] = None,
    ) -> RAGRecommendationResponse:
        """
        Generate ranked, explainable financial product recommendations.

        Parameters
        ----------
        entity_name : Company name (for personalisation).
        query       : Natural language description of financing need.
        top_k       : Max number of recommendations to return.
        categories  : Optional filter — ["lending"], ["grant"], etc.
        amount_eur  : Optional amount filter.
        """
        t0 = time.perf_counter()
        query_id = hashlib.sha256(
            f"{entity_name}:{query}:{time.time()}".encode()
        ).hexdigest()[:12]

        # Route to appropriate agents
        active_categories = categories or ["lending", "grant", "insurance"]
        all_recommendations: List[ProductRecommendation] = []

        for cat in active_categories:
            agent = self._agents.get(cat, self._agents["all"])
            recs = agent.recommend(query, top_k=top_k)
            all_recommendations.extend(recs)

        # Global re-rank by relevance score
        all_recommendations.sort(key=lambda r: r.relevance_score, reverse=True)

        # Apply amount filter
        if amount_eur is not None:
            all_recommendations = [
                r for r in all_recommendations
                if r.max_amount_eur == 0 or r.max_amount_eur >= amount_eur
            ]

        # De-duplicate by product_id
        seen: set = set()
        unique_recs: List[ProductRecommendation] = []
        for r in all_recommendations:
            if r.product_id not in seen:
                seen.add(r.product_id)
                unique_recs.append(r)

        top_recs = unique_recs[:top_k]

        # Assemble RAG context (for LLM / audit log)
        rag_context = self._assemble_rag_context(entity_name, query, top_recs)

        elapsed_ms = (time.perf_counter() - t0) * 1000

        return RAGRecommendationResponse(
            query_id=query_id,
            entity_name=entity_name,
            query=query,
            recommendations=top_recs,
            rag_context=rag_context,
            total_products_evaluated=len(PRODUCT_CATALOGUE),
            processing_time_ms=round(elapsed_ms, 2),
            model_used=self._model_name,
            generated_at=datetime.utcnow().isoformat(),
            eu_ai_act_transparency=self.EU_AI_ACT_DISCLOSURE,
        )

    @staticmethod
    def _assemble_rag_context(
        entity_name: str,
        query: str,
        recommendations: List[ProductRecommendation],
    ) -> str:
        """Build a structured context block for downstream LLM consumption."""
        lines = [
            f"SME: {entity_name}",
            f"Query: {query}",
            f"Top {len(recommendations)} recommended products:",
            "",
        ]
        for i, rec in enumerate(recommendations, 1):
            lines.append(
                f"{i}. [{rec.product_id}] {rec.product_name} "
                f"(score={rec.relevance_score:.3f}, fit={rec.fit_assessment})"
            )
            lines.append(f"   Max: EUR {rec.max_amount_eur:,.0f} | Rate: {rec.interest_rate_pct}%")
            lines.append(f"   {rec.explanation}")
            lines.append("")

        return "\n".join(lines)

    def get_product_by_id(self, product_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve full product details by ID."""
        return next((p for p in PRODUCT_CATALOGUE if p["id"] == product_id), None)

    def list_categories(self) -> List[str]:
        return ["lending", "grant", "insurance"]


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    copilot = SMEFinancialCopilot()

    test_queries = [
        ("TechStartup SL", "We need funding for AI R&D and machine learning infrastructure"),
        ("Comercial Frutas SA", "We have seasonal cash flow gaps of 90 days from harvest to payment"),
        ("Energía Verde SL", "We want to install solar panels and EV charging fleet"),
    ]

    for entity, query in test_queries:
        print(f"\n{'='*60}")
        print(f"ENTITY: {entity}")
        print(f"QUERY:  {query}")
        print(f"{'='*60}")
        response = copilot.recommend(entity_name=entity, query=query, top_k=3)
        for rec in response.recommendations:
            print(f"  [{rec.fit_assessment:9s}] {rec.product_name} (score={rec.relevance_score:.3f})")
        print(f"  Processed {response.total_products_evaluated} products in {response.processing_time_ms:.1f}ms")
