import re
import torch
import numpy as np
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional
import structlog

logger = structlog.get_logger()

CASHTAG_RE = re.compile(r'\$([A-Z]{1,5})')


@dataclass
class SentimentResult:
    score: float       # [-1, +1]
    label: str         # positive | neutral | negative
    positive_prob: float
    neutral_prob: float
    negative_prob: float


@dataclass
class NLPResult:
    sentiment: SentimentResult
    tickers: list[str]
    embedding: list[float]


class FinBERTSentiment:
    """finBERT sentiment classifier with batched GPU inference."""

    def __init__(self, model_name: str = "ProsusAI/finbert"):
        from transformers import BertTokenizer, BertForSequenceClassification
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertForSequenceClassification.from_pretrained(model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
        logger.info("finbert_loaded", device=str(self.device))

    def classify(self, text: str) -> SentimentResult:
        return self.classify_batch([text])[0]

    def classify_batch(self, texts: list[str], batch_size: int = 32) -> list[SentimentResult]:
        results = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            encoded = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            ).to(self.device)

            with torch.no_grad():
                logits = self.model(**encoded).logits
                probs = torch.softmax(logits, dim=1).cpu().numpy()

            for p in probs:
                # finBERT label order: positive=0, negative=1, neutral=2
                pos, neg, neu = float(p[0]), float(p[1]), float(p[2])
                score = pos - neg  # ∈ [-1, +1]
                label = "positive" if pos > neg and pos > neu else ("negative" if neg > pos and neg > neu else "neutral")
                results.append(SentimentResult(
                    score=round(score, 4),
                    label=label,
                    positive_prob=pos,
                    neutral_prob=neu,
                    negative_prob=neg,
                ))
        return results


class TickerNERExtractor:
    """
    Two-stage ticker extraction:
    1. Cashtag regex ($AAPL) — O(1) lookup
    2. spaCy NER → company name → ticker lookup table
    """

    def __init__(self, spacy_model: str = "en_core_web_trf"):
        import spacy
        self.nlp = spacy.load(spacy_model)
        self._ticker_lookup = self._load_lookup()

    def _load_lookup(self) -> dict[str, str]:
        """Map company names to tickers. In production, load from DB."""
        return {
            "apple": "AAPL", "microsoft": "MSFT", "google": "GOOGL",
            "alphabet": "GOOGL", "amazon": "AMZN", "meta": "META",
            "tesla": "TSLA", "nvidia": "NVDA", "netflix": "NFLX",
            "berkshire hathaway": "BRK.B", "jpmorgan": "JPM",
            "goldman sachs": "GS", "bank of america": "BAC",
        }

    def extract(self, text: str) -> list[dict]:
        """Returns list of {ticker, confidence}."""
        found: dict[str, float] = {}

        # Stage 1: Cashtags
        for match in CASHTAG_RE.finditer(text):
            found[match.group(1).upper()] = 1.0

        # Stage 2: spaCy NER
        doc = self.nlp(text)
        for ent in doc.ents:
            if ent.label_ in ("ORG", "PRODUCT"):
                name = ent.text.lower().strip()
                if name in self._ticker_lookup:
                    ticker = self._ticker_lookup[name]
                    if ticker not in found:
                        found[ticker] = 0.85

        return [{"ticker": t, "confidence": c} for t, c in found.items()]


class TextEmbedder:
    """SentenceTransformers for semantic similarity."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)
        logger.info("embedder_loaded", model=model_name)

    def encode(self, text: str) -> list[float]:
        embedding = self.model.encode(text, normalize_embeddings=True)
        return embedding.tolist()

    def encode_batch(self, texts: list[str]) -> list[list[float]]:
        embeddings = self.model.encode(texts, normalize_embeddings=True, batch_size=64)
        return embeddings.tolist()


class NLPPipeline:
    """Orchestrates sentiment + NER + embedding in one call."""

    def __init__(self, model_name: str, spacy_model: str, embedding_model: str):
        self.sentiment = FinBERTSentiment(model_name)
        self.ner = TickerNERExtractor(spacy_model)
        self.embedder = TextEmbedder(embedding_model)

    def process(self, text: str) -> NLPResult:
        return NLPResult(
            sentiment=self.sentiment.classify(text),
            tickers=[t["ticker"] for t in self.ner.extract(text)],
            embedding=self.embedder.encode(text),
        )

    def process_batch(self, texts: list[str]) -> list[NLPResult]:
        sentiments = self.sentiment.classify_batch(texts)
        embeddings = self.embedder.encode_batch(texts)
        results = []
        for text, sent, emb in zip(texts, sentiments, embeddings):
            tickers = [t["ticker"] for t in self.ner.extract(text)]
            results.append(NLPResult(sentiment=sent, tickers=tickers, embedding=emb))
        return results


# Singleton — loaded once at app startup
_pipeline: Optional[NLPPipeline] = None


def get_nlp_pipeline() -> NLPPipeline:
    global _pipeline
    if _pipeline is None:
        from core.config import get_settings
        s = get_settings()
        _pipeline = NLPPipeline(s.HUGGINGFACE_MODEL, s.SPACY_MODEL, s.EMBEDDING_MODEL)
    return _pipeline
