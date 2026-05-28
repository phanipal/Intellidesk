"""Tests for KBRetriever: index building, search, and serialization."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.generate_kb import KB_ARTICLES
from src.retriever import KBRetriever, RetrievalResult


@pytest.fixture(scope="module")
def retriever() -> KBRetriever:
    """Build one retriever and reuse; embedding model load is expensive."""
    r = KBRetriever()
    r.build_index(KB_ARTICLES)
    return r


class TestBuildIndex:
    def test_index_built_marker(self, retriever):
        assert retriever.is_built

    def test_index_size_matches_kb(self, retriever):
        assert retriever.n_articles == len(KB_ARTICLES)

    def test_empty_kb_raises(self):
        r = KBRetriever()
        with pytest.raises(ValueError, match="empty"):
            r.build_index([])

    def test_build_from_missing_json_raises(self):
        r = KBRetriever()
        with pytest.raises(FileNotFoundError):
            r.build_from_json(Path("/nonexistent/kb.json"))


class TestSearchContract:
    def test_returns_list_of_results(self, retriever):
        results = retriever.search("VPN connection issue")
        assert isinstance(results, list)
        assert all(isinstance(r, RetrievalResult) for r in results)

    def test_default_top_k_is_3(self, retriever):
        results = retriever.search("VPN issue")
        assert len(results) == 3

    def test_custom_top_k(self, retriever):
        results = retriever.search("VPN issue", top_k=5)
        assert len(results) == 5

    def test_top_k_capped_at_kb_size(self, retriever):
        results = retriever.search("VPN issue", top_k=1000)
        assert len(results) == retriever.n_articles

    def test_results_sorted_by_score_descending(self, retriever):
        results = retriever.search("password reset MFA", top_k=5)
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_scores_in_valid_range(self, retriever):
        """Cosine similarity (on normalized vectors) is in [-1, 1]."""
        results = retriever.search("printer paper jam", top_k=5)
        for r in results:
            assert -1.0 <= r.score <= 1.0

    def test_empty_query_returns_empty(self, retriever):
        assert retriever.search("") == []
        assert retriever.search("   ") == []

    def test_search_before_build_raises(self):
        r = KBRetriever()
        with pytest.raises(RuntimeError, match="Index not built"):
            r.search("any query")


class TestSemanticRelevance:
    """
    Top result for a domain-specific query must be in the right category.
    Acts as the embedding-quality regression guard: if these fail, the
    retriever is returning irrelevant results.
    """

    @pytest.mark.parametrize("query,expected_category", [
        ("Cannot connect to corporate VPN from home", "Network"),
        ("Office WiFi keeps dropping during meetings", "Network"),
        ("Outlook crashes every morning on launch", "Software"),
        ("Software Center install failed for me", "Software"),
        ("My laptop battery wont charge anymore", "Hardware"),
        ("Printer is jammed and not printing", "Hardware"),
        ("Locked out of my account, need password reset", "Access"),
        ("SSO failing globally, no one can log in", "Access"),
    ])
    def test_top_result_matches_expected_category(
        self, retriever, query, expected_category
    ):
        results = retriever.search(query, top_k=1)
        assert len(results) == 1, f"Expected 1 result for '{query}'"
        assert results[0].category == expected_category, (
            f"Query '{query}' expected {expected_category}, got "
            f"{results[0].kb_id} ({results[0].category})"
        )

    def test_outage_query_finds_outage_kb(self, retriever):
        """Major-outage language should retrieve at least one outage-handling KB."""
        results = retriever.search(
            "entire site is down, multiple users affected — major incident",
            top_k=3,
        )
        outage_kbs = {"KB-NET-003", "KB-SW-004", "KB-ACC-003"}
        top_ids = {r.kb_id for r in results}
        assert top_ids & outage_kbs, (
            f"Expected at least one outage KB in top 3, got {top_ids}"
        )


class TestBatchSearch:
    def test_batch_returns_list_per_query(self, retriever):
        queries = ["VPN issue", "Outlook crash", "Password reset"]
        results = retriever.search_batch(queries, top_k=2)
        assert len(results) == 3
        for r in results:
            assert len(r) == 2

    def test_batch_matches_single_search(self, retriever):
        queries = [
            "VPN connection failure",
            "Outlook keeps crashing",
            "Need password reset",
        ]
        single = [retriever.search(q, top_k=2) for q in queries]
        batched = retriever.search_batch(queries, top_k=2)
        for s, b in zip(single, batched):
            assert [r.kb_id for r in s] == [r.kb_id for r in b]

    def test_batch_handles_mixed_empty_queries(self, retriever):
        results = retriever.search_batch(
            ["VPN issue", "", "  ", "Outlook crash"], top_k=2
        )
        assert len(results) == 4
        assert results[0]
        assert results[1] == []
        assert results[2] == []
        assert results[3]


class TestDeterminism:
    def test_same_query_same_results(self, retriever):
        q = "VPN is not connecting"
        r1 = retriever.search(q)
        r2 = retriever.search(q)
        assert [(r.kb_id, round(r.score, 6)) for r in r1] == \
               [(r.kb_id, round(r.score, 6)) for r in r2]


class TestPersistence:
    def test_save_creates_files(self, retriever, tmp_path):
        save_dir = tmp_path / "retriever_test"
        retriever.save(save_dir)
        assert (save_dir / "faiss.index").exists()
        assert (save_dir / "metadata.json").exists()

    def test_load_restores_search_results_exactly(self, retriever, tmp_path):
        save_dir = tmp_path / "retriever_test"
        retriever.save(save_dir)

        loaded = KBRetriever.load(save_dir)
        assert loaded.is_built
        assert loaded.n_articles == retriever.n_articles

        q = "Major SSO outage affecting our department"
        before = retriever.search(q, top_k=3)
        after = loaded.search(q, top_k=3)
        assert [r.kb_id for r in before] == [r.kb_id for r in after]
        for b, a in zip(before, after):
            assert abs(b.score - a.score) < 1e-5

    def test_save_unbuilt_raises(self):
        r = KBRetriever()
        with pytest.raises(RuntimeError, match="unbuilt"):
            r.save()

    def test_load_missing_dir_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            KBRetriever.load(tmp_path / "nonexistent")
