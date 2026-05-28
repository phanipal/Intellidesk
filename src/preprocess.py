"""
Single source of truth for ticket text preprocessing.

The TicketPreprocessor produces TWO flavors of cleaned text per ticket:
  - tfidf_text:  aggressive cleaning (lowercase, lemmatized, stopwords removed)
                 for XGBoost + TF-IDF baseline
  - raw_text:    minimal cleaning (PII scrubbed only) for sentence-transformer
                 embeddings, which work best on natural sentence structure

Used by:
  - notebooks/02_modeling.ipynb (training)
  - src/classifier.py (training + inference)
  - src/retriever.py (embedding KB articles + queries)
  - src/api.py (live request preprocessing)
  - dashboard/app.py (preview cleaning effect on tickets)

Funneling every caller through this class avoids train/serve skew.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Iterable, List, Optional

import pandas as pd

from src.config import DEFAULT_CONFIG, PreprocessConfig

logger = logging.getLogger("intellidesk.preprocess")


# Order matters: more specific patterns must match before generic ones.
# Each pattern maps to a placeholder token the model can learn as a feature.
PII_PATTERNS = [
    (re.compile(r"\b[\w._%+-]+@[\w.-]+\.[A-Za-z]{2,}\b"), " <EMAIL> "),
    (re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"), " <IP> "),
    (re.compile(r"https?://\S+"), " <URL> "),
    # Hostnames like server01.corp.local, host-prod-db
    (re.compile(r"\b[a-zA-Z][a-zA-Z0-9-]*\.(corp|local|internal|net|com)\b"), " <HOST> "),
    # Employee/user IDs like U12345, E98765
    (re.compile(r"\b[UE]\d{4,6}\b"), " <USERID> "),
    # Ticket IDs like INC10000123
    (re.compile(r"\bINC\d{6,10}\b"), " <TICKETID> "),
    (re.compile(r"\b([0-9A-Fa-f]{2}[:-]){5}[0-9A-Fa-f]{2}\b"), " <MAC> "),
    # Phone numbers (US-ish formats)
    (re.compile(r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"), " <PHONE> "),
]

WS_RE = re.compile(r"\s+")


@dataclass(frozen=True)
class PreprocessedTicket:
    """Result of preprocessing one ticket description."""
    raw_text: str        # PII-scrubbed only, for sentence-transformer
    tfidf_text: str      # fully cleaned + lemmatized, for TF-IDF/XGBoost
    token_count: int


class TicketPreprocessor:
    """
    Stateless preprocessor for IT ticket descriptions.

    Lazy-loads spaCy on first use so importing the module is cheap (matters
    for FastAPI cold starts and notebook responsiveness).

    Usage:
        prep = TicketPreprocessor()
        result = prep.process_one("Outlook crashed at user@corp.com")
        # result.raw_text   -> "Outlook crashed at <EMAIL>"
        # result.tfidf_text -> "outlook crash email"

        df = prep.process_dataframe(tickets_df, text_col="description")
        # adds 'raw_text', 'tfidf_text', 'token_count' columns
    """

    def __init__(self, config: Optional[PreprocessConfig] = None):
        self.config = config or DEFAULT_CONFIG
        self._nlp = None
        self._stopwords: Optional[frozenset] = None

    def _load_spacy(self):
        """Load spaCy model on first use. Raises a clear error if missing."""
        if self._nlp is not None:
            return self._nlp

        try:
            import spacy
        except ImportError as exc:
            raise ImportError(
                "spaCy is required. Install with: pip install spacy"
            ) from exc

        try:
            self._nlp = spacy.load(
                self.config.spacy_model,
                disable=list(self.config.spacy_disable),
            )
        except OSError as exc:
            raise OSError(
                f"spaCy model '{self.config.spacy_model}' not found. "
                f"Install with: python -m spacy download {self.config.spacy_model}"
            ) from exc

        if self.config.remove_stopwords:
            base_stops = self._nlp.Defaults.stop_words
            keep = self.config.keep_domain_terms or frozenset()
            self._stopwords = frozenset(base_stops - keep)
            logger.info(
                "spaCy loaded with stopword filtering: %s "
                "(%d stopwords, %d keep-terms)",
                self.config.spacy_model, len(self._stopwords), len(keep),
            )
        else:
            self._stopwords = None
            logger.info(
                "spaCy loaded, stopword filtering disabled: %s",
                self.config.spacy_model,
            )

        return self._nlp

    @staticmethod
    def _scrub_pii(text: str) -> str:
        """Replace PII with placeholder tokens. Idempotent."""
        if not text:
            return ""
        for pattern, replacement in PII_PATTERNS:
            text = pattern.sub(replacement, text)
        return WS_RE.sub(" ", text).strip()

    def _clean_for_tfidf(self, doc) -> str:
        """
        Lowercase + lemmatize + drop stopwords/punct/short tokens.
        Keeps PII placeholder tokens (<EMAIL> etc.) intact as features.
        """
        cfg = self.config
        kept = []
        for token in doc:
            # PII placeholders carry signal, preserve them
            text = token.text
            if text.startswith("<") and text.endswith(">"):
                kept.append(text.lower() if cfg.lowercase else text)
                continue

            if token.is_punct or token.is_space:
                continue
            if cfg.remove_stopwords and self._stopwords and token.text.lower() in self._stopwords:
                continue

            lemma = token.lemma_ if cfg.lemmatize else token.text
            if cfg.lowercase:
                lemma = lemma.lower()

            if len(lemma) < cfg.min_token_length:
                continue

            kept.append(lemma)
        return " ".join(kept)

    def process_one(self, text: str) -> PreprocessedTicket:
        """Process a single ticket description."""
        if text is None or (isinstance(text, float) and pd.isna(text)):
            return PreprocessedTicket(raw_text="", tfidf_text="", token_count=0)

        scrubbed = self._scrub_pii(str(text)) if self.config.strip_pii else str(text)
        nlp = self._load_spacy()
        doc = nlp(scrubbed)
        cleaned = self._clean_for_tfidf(doc)
        return PreprocessedTicket(
            raw_text=scrubbed,
            tfidf_text=cleaned,
            token_count=len(cleaned.split()) if cleaned else 0,
        )

    def process_batch(
        self,
        texts: Iterable[str],
        batch_size: int = 256,
    ) -> List[PreprocessedTicket]:
        """Process many ticket descriptions efficiently using spaCy's pipe."""
        texts_list = ["" if t is None or (isinstance(t, float) and pd.isna(t)) else str(t)
                      for t in texts]

        scrubbed_list = (
            [self._scrub_pii(t) for t in texts_list]
            if self.config.strip_pii else texts_list
        )

        nlp = self._load_spacy()
        results: List[PreprocessedTicket] = []
        for scrubbed, doc in zip(scrubbed_list, nlp.pipe(scrubbed_list, batch_size=batch_size)):
            cleaned = self._clean_for_tfidf(doc)
            results.append(PreprocessedTicket(
                raw_text=scrubbed,
                tfidf_text=cleaned,
                token_count=len(cleaned.split()) if cleaned else 0,
            ))
        return results

    def process_dataframe(
        self,
        df: pd.DataFrame,
        text_col: str = "description",
        batch_size: int = 256,
    ) -> pd.DataFrame:
        """
        Add raw_text, tfidf_text, and token_count columns to a DataFrame.
        Returns a new DataFrame; does not mutate the input.
        """
        if text_col not in df.columns:
            raise KeyError(f"Column '{text_col}' not in DataFrame. "
                           f"Available: {list(df.columns)}")

        logger.info("Preprocessing %d tickets (batch_size=%d)", len(df), batch_size)
        results = self.process_batch(df[text_col].tolist(), batch_size=batch_size)

        out = df.copy()
        out["raw_text"] = [r.raw_text for r in results]
        out["tfidf_text"] = [r.tfidf_text for r in results]
        out["token_count"] = [r.token_count for r in results]
        return out


def main() -> None:
    """CLI: preprocess data/tickets.csv -> tmp/tickets_preprocessed.csv"""
    import argparse

    from src.config import TICKETS_CSV, TMP_DIR

    parser = argparse.ArgumentParser(description="Preprocess ticket descriptions.")
    parser.add_argument("--in", dest="input_path", type=str, default=str(TICKETS_CSV))
    parser.add_argument("--out", dest="output_path", type=str,
                        default=str(TMP_DIR / "tickets_preprocessed.csv"))
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    df = pd.read_csv(args.input_path)
    prep = TicketPreprocessor()
    out_df = prep.process_dataframe(df, text_col="description")

    from pathlib import Path
    Path(args.output_path).parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.output_path, index=False)
    logger.info("Saved preprocessed CSV → %s", args.output_path)


if __name__ == "__main__":
    main()
