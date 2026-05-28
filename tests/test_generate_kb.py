"""
Unit tests for the knowledge base generator. Verifies the KB ↔ ticket-template
alignment so the retriever has something valid to return for every ticket type.

Run:
    pytest tests/test_generate_kb.py -v
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.generate_kb import KB_ARTICLES
from src.generate_data import TEMPLATES

REQUIRED_FIELDS = {"kb_id", "title", "category", "tags", "content"}
VALID_CATEGORIES = {"Network", "Software", "Hardware", "Access"}


def test_kb_is_nonempty():
    assert len(KB_ARTICLES) > 0


def test_every_article_has_required_fields():
    for art in KB_ARTICLES:
        missing = REQUIRED_FIELDS - art.keys()
        assert not missing, f"{art.get('kb_id')} missing fields: {missing}"


def test_kb_ids_unique():
    ids = [a["kb_id"] for a in KB_ARTICLES]
    assert len(ids) == len(set(ids))


def test_kb_id_format():
    """IDs follow KB-{CAT}-{NNN} pattern."""
    import re
    pattern = re.compile(r"^KB-(NET|SW|HW|ACC)-\d{3}$")
    for art in KB_ARTICLES:
        assert pattern.match(art["kb_id"]), f"Bad ID: {art['kb_id']}"


def test_categories_valid():
    for art in KB_ARTICLES:
        assert art["category"] in VALID_CATEGORIES


def test_content_substantive():
    """Content must be long enough for embeddings to be meaningful."""
    for art in KB_ARTICLES:
        assert len(art["content"]) >= 100, f"{art['kb_id']} too short"


def test_tags_are_list_of_strings():
    for art in KB_ARTICLES:
        assert isinstance(art["tags"], list)
        assert all(isinstance(t, str) for t in art["tags"])
        assert len(art["tags"]) > 0


def test_every_template_has_matching_kb_article():
    """
    Every TicketTemplate references a KB article via resolution_kb_id. The
    retriever depends on this; orphan references would silently return empty.
    """
    kb_ids = {a["kb_id"] for a in KB_ARTICLES}
    template_refs = {t.resolution_kb_id for t in TEMPLATES}
    orphans = template_refs - kb_ids
    assert not orphans, f"Templates reference missing KB articles: {orphans}"


def test_kb_categories_match_referencing_templates():
    """A KB article's category should match the templates that reference it."""
    kb_by_id = {a["kb_id"]: a for a in KB_ARTICLES}
    for t in TEMPLATES:
        kb = kb_by_id[t.resolution_kb_id]
        assert kb["category"] == t.category, (
            f"Template {t.subcategory} ({t.category}) references "
            f"{kb['kb_id']} which is in {kb['category']}"
        )


def test_kb_file_writes_valid_json(tmp_path, monkeypatch):
    """Run main() into a temp dir and verify the output parses."""
    monkeypatch.chdir(tmp_path)
    from src import generate_kb
    generate_kb.main()

    out = tmp_path / "data" / "knowledge_base.json"
    assert out.exists()
    loaded = json.loads(out.read_text())
    assert loaded == KB_ARTICLES
