"""Tests for TicketTriagePipeline: end-to-end triage, review flags, and batch processing."""

from __future__ import annotations

import json

import pytest

from src.classifier import TicketClassifier
from src.config import CATEGORIES, PRIORITIES
from src.pipeline import (
    TicketTriagePipeline,
    TriageConfig,
    TriageResult,
)
from src.retriever import KBRetriever


def _artifacts_exist() -> bool:
    """Skip pipeline tests if saved model/index don't exist yet."""
    return (
        TicketClassifier.DEFAULT_MODEL_PATH.exists()
        and (KBRetriever.DEFAULT_INDEX_DIR / KBRetriever.INDEX_FILE).exists()
    )


pytestmark = pytest.mark.skipif(
    not _artifacts_exist(),
    reason="Pipeline tests need saved artifacts. Run "
           "'.\\make.ps1 train-quick' and '.\\make.ps1 build-index' first."
)


@pytest.fixture(scope="module")
def pipeline() -> TicketTriagePipeline:
    """Single warmed-up pipeline reused across tests."""
    p = TicketTriagePipeline()
    p.warm_up()
    return p


class TestLoading:
    def test_pipeline_not_ready_before_warmup(self):
        p = TicketTriagePipeline()
        assert not p.is_ready

    def test_pipeline_ready_after_warmup(self, pipeline):
        assert pipeline.is_ready

    def test_warmup_is_idempotent(self):
        p = TicketTriagePipeline()
        p.warm_up()
        p.warm_up()  # second call should not error or reload
        assert p.is_ready


class TestTriageContract:
    def test_returns_triage_result(self, pipeline):
        result = pipeline.triage("VPN is down for entire team")
        assert isinstance(result, TriageResult)

    def test_category_in_valid_set(self, pipeline):
        result = pipeline.triage("Outlook keeps crashing")
        assert result.category in CATEGORIES

    def test_priority_in_valid_set(self, pipeline):
        result = pipeline.triage("Need urgent help")
        assert result.priority in PRIORITIES

    def test_confidences_in_unit_interval(self, pipeline):
        result = pipeline.triage("Password reset needed")
        assert 0.0 <= result.category_confidence <= 1.0
        assert 0.0 <= result.priority_confidence <= 1.0

    def test_latency_recorded_and_positive(self, pipeline):
        result = pipeline.triage("WiFi is slow")
        assert result.latency_ms > 0

    def test_ticket_text_echoed_back(self, pipeline):
        text = "Specific test ticket text here"
        result = pipeline.triage(text)
        assert result.ticket_text == text

    def test_to_dict_is_json_serializable(self, pipeline):
        """FastAPI requires the response to serialize to JSON."""
        result = pipeline.triage("VPN issue")
        d = result.to_dict()
        json.dumps(d)  # raises TypeError if not serializable


class TestKBSuggestions:
    def test_kb_suggestions_returned_for_clear_query(self, pipeline):
        result = pipeline.triage("VPN tunnel keeps dropping every few minutes")
        assert isinstance(result.kb_suggestions, list)
        assert len(result.kb_suggestions) > 0

    def test_kb_suggestions_have_required_fields(self, pipeline):
        result = pipeline.triage("Major SSO outage in finance")
        for kb in result.kb_suggestions:
            assert "kb_id" in kb
            assert "title" in kb
            assert "category" in kb
            assert "score" in kb

    def test_kb_filtered_by_min_score(self, pipeline):
        result = pipeline.triage("VPN issue")
        for kb in result.kb_suggestions:
            assert kb["score"] >= pipeline.config.min_kb_score

    def test_top_k_override(self, pipeline):
        result_2 = pipeline.triage("VPN issue", top_k=2)
        assert len(result_2.kb_suggestions) <= 2


class TestHumanReviewRouting:
    def test_low_confidence_triggers_review(self):
        """Force review by setting unrealistically high thresholds."""
        config = TriageConfig(
            category_confidence_threshold=0.99,
            priority_confidence_threshold=0.99,
        )
        p = TicketTriagePipeline(config=config)
        p.warm_up()
        result = p.triage("Some ambiguous ticket text here")
        assert result.needs_human_review
        assert any("category confidence" in r for r in result.review_reasons)
        assert any("priority confidence" in r for r in result.review_reasons)

    def test_high_confidence_skips_review(self):
        """Force no-review by setting trivially low thresholds."""
        config = TriageConfig(
            category_confidence_threshold=0.01,
            priority_confidence_threshold=0.01,
            min_kb_score=0.0,
        )
        p = TicketTriagePipeline(config=config)
        p.warm_up()
        result = p.triage("VPN tunnel disconnecting from corporate network")
        assert not result.needs_human_review
        assert result.review_reasons == []

    def test_review_reasons_explain_why(self):
        config = TriageConfig(category_confidence_threshold=0.99)
        p = TicketTriagePipeline(config=config)
        p.warm_up()
        result = p.triage("Some ticket")
        if result.needs_human_review:
            assert len(result.review_reasons) > 0


class TestEdgeCases:
    def test_empty_string_handled(self, pipeline):
        result = pipeline.triage("")
        assert isinstance(result, TriageResult)
        assert result.category in CATEGORIES

    def test_none_handled(self, pipeline):
        result = pipeline.triage(None)
        assert isinstance(result, TriageResult)

    def test_very_long_text_handled(self, pipeline):
        long_text = "VPN is down. " * 500
        result = pipeline.triage(long_text)
        assert isinstance(result, TriageResult)


class TestBatch:
    def test_batch_returns_correct_count(self, pipeline):
        texts = ["VPN down", "Outlook crash", "Password reset"]
        results = pipeline.triage_batch(texts)
        assert len(results) == 3

    def test_batch_results_match_single(self, pipeline):
        """Batch must produce identical results to per-ticket calls."""
        texts = [
            "VPN tunnel keeps dropping",
            "Outlook keeps crashing",
            "Major SSO outage in finance team",
        ]
        single = [pipeline.triage(t) for t in texts]
        batch = pipeline.triage_batch(texts)
        for s, b in zip(single, batch):
            assert s.category == b.category
            assert s.priority == b.priority
            assert [k["kb_id"] for k in s.kb_suggestions] == \
                   [k["kb_id"] for k in b.kb_suggestions]

    def test_batch_handles_empty_list(self, pipeline):
        assert pipeline.triage_batch([]) == []
