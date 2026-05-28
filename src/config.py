"""
Central configuration for IntelliDesk. All paths, hyperparameters, and
constants live here so other modules don't hardcode values.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import FrozenSet

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
DATA_DIR: Path = PROJECT_ROOT / "data"
MODELS_DIR: Path = PROJECT_ROOT / "models"
REPORTS_DIR: Path = PROJECT_ROOT / "reports"
TMP_DIR: Path = PROJECT_ROOT / "tmp"

TICKETS_CSV: Path = DATA_DIR / "tickets.csv"
KB_JSON: Path = DATA_DIR / "knowledge_base.json"

CATEGORIES: tuple = ("Network", "Software", "Hardware", "Access")
PRIORITIES: tuple = ("P1", "P2", "P3", "P4")


@dataclass(frozen=True)
class PreprocessConfig:
    """
    Preprocessing configuration.

    Defaults:
      - lowercase + lemmatize
      - PII scrubbing on
      - stopword removal OFF: TfidfVectorizer's max_df handles stopwords
        statistically downstream, which avoids brittle manual exception
        lists that go stale as the corpus evolves.

    Stopword removal is available as an opt-in (remove_stopwords=True). In
    that case, supply your own keep_domain_terms set to protect specific
    vocabulary.
    """
    lowercase: bool = True
    strip_pii: bool = True
    remove_stopwords: bool = False  # off by default, see docstring
    lemmatize: bool = True
    min_token_length: int = 2
    keep_domain_terms: FrozenSet[str] = field(default_factory=frozenset)
    spacy_model: str = "en_core_web_sm"
    spacy_disable: tuple = ("parser", "ner")  # speeds up ~3x


DEFAULT_CONFIG = PreprocessConfig()
