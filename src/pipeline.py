"""
Unified ticket triage pipeline.

Combines the trained classifier and KB retriever into one triage() call.
Used by the API, dashboard, CLI, and validation suite.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional

from src.classifier import Prediction, TicketClassifier
from src.retriever import KBRetriever, RetrievalResult

logger = logging.getLogger("intellidesk.pipeline")


@dataclass(frozen=True)
class TriageConfig:
    """
    Configuration for triage decisions.

    Confidence thresholds determine when to flag a ticket for human review.
    Below these, the pipeline marks needs_human_review=True so the API/
    dashboard can route low-confidence tickets to a queue instead of
    auto-actioning. Tune empirically based on org's error tolerance.
    """
    top_k: int = 3
    category_confidence_threshold: float = 0.50
    priority_confidence_threshold: float = 0.40
    min_kb_score: float = 0.30  # drop KB suggestions below this similarity


DEFAULT_CONFIG = TriageConfig()


@dataclass(frozen=True)
class TriageResult:
    """
    Complete triage decision for one ticket.

    FastAPI serializes this directly to JSON and the dashboard renders it
    as cards. Adding or removing fields here is a contract change.
    """
    ticket_text: str

    category: str
    category_confidence: float
    priority: str
    priority_confidence: float

    kb_suggestions: List[Dict]  # serialized RetrievalResult dicts

    needs_human_review: bool
    review_reasons: List[str]

    latency_ms: float

    def to_dict(self) -> Dict:
        return asdict(self)


class TicketTriagePipeline:
    """
    Orchestrates classification and retrieval for ticket triage.

    Components are lazy-loaded and reused across all subsequent calls.
    """

    def __init__(
        self,
        classifier_path: Optional[Path] = None,
        retriever_dir: Optional[Path] = None,
        config: Optional[TriageConfig] = None,
    ):
        self._classifier_path = classifier_path
        self._retriever_dir = retriever_dir
        self._config = config or DEFAULT_CONFIG
        self._classifier: Optional[TicketClassifier] = None
        self._retriever: Optional[KBRetriever] = None

    @property
    def is_ready(self) -> bool:
        """True once both classifier and retriever are loaded."""
        return self._classifier is not None and self._retriever is not None

    @property
    def config(self) -> TriageConfig:
        return self._config

    def _load_classifier(self) -> TicketClassifier:
        if self._classifier is None:
            path = self._classifier_path or TicketClassifier.DEFAULT_MODEL_PATH
            logger.info("Loading classifier from %s", path)
            self._classifier = TicketClassifier.load(self._classifier_path)
        return self._classifier

    def _load_retriever(self) -> KBRetriever:
        if self._retriever is None:
            path = self._retriever_dir or KBRetriever.DEFAULT_INDEX_DIR
            logger.info("Loading retriever from %s", path)
            self._retriever = KBRetriever.load(self._retriever_dir)
        return self._retriever

    def warm_up(self) -> None:
        """Eagerly load classifier and retriever."""
        self._load_classifier()
        self._load_retriever()
        logger.info("Pipeline warmed up — ready for triage requests")

    def triage(self, text: str, top_k: Optional[int] = None) -> TriageResult:
        """
        Triage a single ticket: classify and retrieve KB articles in one call.

        Args:
            text: Raw ticket description (preprocessing applied internally).
            top_k: Override config's default top_k for this call.

        Returns:
            TriageResult with classification, KB suggestions, and review flags.
        """
        start = time.time()

        text = text or ""
        k = top_k if top_k is not None else self._config.top_k

        classifier = self._load_classifier()
        prediction = classifier.predict(text)

        # filter low-similarity hits to avoid junk suggestions
        retriever = self._load_retriever()
        kb_results = retriever.search(text, top_k=k)
        kb_filtered = [r for r in kb_results if r.score >= self._config.min_kb_score]

        review_reasons = self._determine_review_reasons(prediction, kb_filtered)

        latency_ms = (time.time() - start) * 1000.0

        return TriageResult(
            ticket_text=text,
            category=prediction.category,
            category_confidence=prediction.category_confidence,
            priority=prediction.priority,
            priority_confidence=prediction.priority_confidence,
            kb_suggestions=[r.to_dict() for r in kb_filtered],
            needs_human_review=bool(review_reasons),
            review_reasons=review_reasons,
            latency_ms=round(latency_ms, 2),
        )

    def triage_batch(
        self,
        texts: List[str],
        top_k: Optional[int] = None,
    ) -> List[TriageResult]:
        """Triage multiple tickets in one call."""
        if not texts:
            return []

        start = time.time()

        texts = [t or "" for t in texts]
        k = top_k if top_k is not None else self._config.top_k

        classifier = self._load_classifier()
        predictions = classifier.predict_batch(texts)

        retriever = self._load_retriever()
        kb_results_list = retriever.search_batch(texts, top_k=k)

        results: List[TriageResult] = []
        for text, pred, kb_results in zip(texts, predictions, kb_results_list):
            kb_filtered = [r for r in kb_results if r.score >= self._config.min_kb_score]
            review_reasons = self._determine_review_reasons(pred, kb_filtered)
            results.append(TriageResult(
                ticket_text=text,
                category=pred.category,
                category_confidence=pred.category_confidence,
                priority=pred.priority,
                priority_confidence=pred.priority_confidence,
                kb_suggestions=[r.to_dict() for r in kb_filtered],
                needs_human_review=bool(review_reasons),
                review_reasons=review_reasons,
                latency_ms=0.0,  # individual latency not meaningful in batch
            ))

        total_latency = (time.time() - start) * 1000.0
        logger.info(
            "Batch triage: %d tickets in %.0fms (%.1fms/ticket avg)",
            len(texts), total_latency, total_latency / max(len(texts), 1),
        )
        return results

    def _determine_review_reasons(
        self,
        prediction: Prediction,
        kb_results: List[RetrievalResult],
    ) -> List[str]:
        """Decide whether this ticket needs human review and return the reasons."""
        reasons = []
        if prediction.category_confidence < self._config.category_confidence_threshold:
            reasons.append(
                f"low category confidence ({prediction.category_confidence:.2f} < "
                f"{self._config.category_confidence_threshold})"
            )
        if prediction.priority_confidence < self._config.priority_confidence_threshold:
            reasons.append(
                f"low priority confidence ({prediction.priority_confidence:.2f} < "
                f"{self._config.priority_confidence_threshold})"
            )
        if not kb_results:
            reasons.append("no KB articles above similarity threshold")
        return reasons


def main() -> None:
    """CLI: triage one or more tickets from command line."""
    import argparse
    import json
    import sys

    parser = argparse.ArgumentParser(
        description="Triage IT tickets through the full pipeline."
    )
    parser.add_argument(
        "text", nargs="?",
        help="Ticket description. If omitted, reads stdin or runs samples."
    )
    parser.add_argument("--top-k", type=int, default=3,
                        help="Number of KB suggestions to return")
    parser.add_argument("--json", action="store_true",
                        help="Output as JSON instead of human-readable")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    pipeline = TicketTriagePipeline()
    pipeline.warm_up()

    if args.text:
        texts = [args.text]
    elif not sys.stdin.isatty():
        texts = [line.strip() for line in sys.stdin if line.strip()]
    else:
        texts = [
            "VPN keeps disconnecting every 10 minutes from home office",
            "Outlook crashes when opening large email attachments",
            "Locked out of my AD account after too many login attempts",
            "Major SSO outage affecting our entire finance team this morning",
        ]
        print("\nNo input provided — running sample tickets...\n")

    for text in texts:
        result = pipeline.triage(text, top_k=args.top_k)
        if args.json:
            print(json.dumps(result.to_dict(), indent=2))
        else:
            _print_human_readable(result)


def _print_human_readable(result: TriageResult) -> None:
    """Pretty-print one triage result to stdout."""
    print("\n" + "=" * 70)
    print(f"  TICKET:   {result.ticket_text}")
    print("-" * 70)
    print(f"  CATEGORY: {result.category:10s} (confidence {result.category_confidence:.2f})")
    print(f"  PRIORITY: {result.priority:10s} (confidence {result.priority_confidence:.2f})")
    print(f"  LATENCY:  {result.latency_ms} ms")
    if result.needs_human_review:
        print(f"  REVIEW:   YES")
        for reason in result.review_reasons:
            print(f"            - {reason}")
    else:
        print(f"  REVIEW:   no (auto-handle eligible)")
    print(f"  KB SUGGESTIONS:")
    if not result.kb_suggestions:
        print("    (none above similarity threshold)")
    for i, kb in enumerate(result.kb_suggestions, 1):
        print(f"    {i}. [{kb['kb_id']}] {kb['title']}")
        print(f"       Score {kb['score']:.3f}  |  Category: {kb['category']}")
    print("=" * 70)


if __name__ == "__main__":
    main()
