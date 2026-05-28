"""Tests for TicketPreprocessor: PII scrubbing, lemmatization, and batch processing."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.config import PreprocessConfig
from src.preprocess import (
    PII_PATTERNS,
    PreprocessedTicket,
    TicketPreprocessor,
)


def _ensure_spacy_model() -> None:
    """Auto-download the spaCy model if missing. Runs once per session."""
    import subprocess
    import sys
    try:
        import spacy
        spacy.load("en_core_web_sm")
    except (ImportError, OSError):
        print("\n[setup] Downloading spaCy en_core_web_sm model (one-time)...")
        subprocess.run(
            [sys.executable, "-m", "spacy", "download", "en_core_web_sm"],
            check=True,
        )

_ensure_spacy_model()


@pytest.fixture(scope="module")
def prep() -> TicketPreprocessor:
    """Default preprocessor; stopwords kept."""
    return TicketPreprocessor()


@pytest.fixture(scope="module")
def prep_with_stopword_removal() -> TicketPreprocessor:
    """Opt-in preprocessor with stopword filtering enabled."""
    return TicketPreprocessor(PreprocessConfig(remove_stopwords=True))


class TestPIIScrubbing:
    def test_email_scrubbed(self, prep):
        result = prep.process_one("Email me at john.doe@corp.com please")
        assert "<EMAIL>" in result.raw_text
        assert "john.doe@corp.com" not in result.raw_text

    def test_ip_scrubbed(self, prep):
        result = prep.process_one("Server at 192.168.1.42 is down")
        assert "<IP>" in result.raw_text
        assert "192.168.1.42" not in result.raw_text

    def test_url_scrubbed(self, prep):
        result = prep.process_one("Check https://internal.corp/wiki for details")
        assert "<URL>" in result.raw_text
        assert "https://" not in result.raw_text

    def test_hostname_scrubbed(self, prep):
        result = prep.process_one("Server prod-db01.corp is unreachable")
        assert "<HOST>" in result.raw_text

    def test_userid_scrubbed(self, prep):
        result = prep.process_one("Created by U12345 last week")
        assert "<USERID>" in result.raw_text
        assert "U12345" not in result.raw_text

    def test_ticketid_scrubbed(self, prep):
        result = prep.process_one("Related to INC10000042 from yesterday")
        assert "<TICKETID>" in result.raw_text

    def test_mac_address_scrubbed(self, prep):
        result = prep.process_one("MAC 00:1A:2B:3C:4D:5E not registering")
        assert "<MAC>" in result.raw_text

    def test_phone_scrubbed(self, prep):
        result = prep.process_one("Call me at (555) 123-4567 to confirm")
        assert "<PHONE>" in result.raw_text

    def test_multiple_pii_in_one_text(self, prep):
        result = prep.process_one(
            "User U99999 (jane@corp.com) reported 10.0.0.5 unreachable"
        )
        assert "<USERID>" in result.raw_text
        assert "<EMAIL>" in result.raw_text
        assert "<IP>" in result.raw_text

    def test_pii_scrubbing_is_idempotent(self, prep):
        text = "Email user@corp.com about server 192.168.1.1"
        once = prep._scrub_pii(text)
        twice = prep._scrub_pii(once)
        assert once == twice

    def test_pii_disabled_by_config(self):
        no_pii_prep = TicketPreprocessor(PreprocessConfig(strip_pii=False))
        result = no_pii_prep.process_one("user@corp.com")
        assert "user@corp.com" in result.raw_text


class TestTfidfCleaning:
    def test_lowercase_applied(self, prep):
        result = prep.process_one("OUTLOOK CRASH ON LAUNCH")
        assert result.tfidf_text == result.tfidf_text.lower()

    def test_punctuation_removed(self, prep):
        result = prep.process_one("VPN: down! Help, please.")
        assert ":" not in result.tfidf_text
        assert "!" not in result.tfidf_text

    def test_lemmatization_applied(self, prep):
        result = prep.process_one("Servers are crashing repeatedly")
        # "crashing" -> "crash", "servers" -> "server"
        assert "crash" in result.tfidf_text
        assert "server" in result.tfidf_text

    def test_pii_placeholders_kept_in_tfidf(self, prep):
        result = prep.process_one("Email user@corp.com about issue")
        assert "<email>" in result.tfidf_text or "email" in result.tfidf_text

    def test_short_tokens_filtered(self):
        """min_token_length=3 should drop 1- and 2-char tokens."""
        cfg = PreprocessConfig(min_token_length=3)
        prep = TicketPreprocessor(cfg)
        result = prep.process_one("a bb ccc dddd")
        tokens = result.tfidf_text.split()
        assert "a" not in tokens
        assert "bb" not in tokens
        assert "ccc" in tokens
        assert "dddd" in tokens


class TestStopwordsDefault:
    """Default behavior keeps stopwords; TfidfVectorizer handles them."""

    def test_stopwords_preserved_by_default(self, prep):
        """
        Stopwords whose lemma equals themselves ('the', 'and', 'on') should
        survive default preprocessing. We deliberately use words that aren't
        affected by lemmatization here so the test isolates one behavior.
        """
        result = prep.process_one("the laptop and the desk on the floor")
        tokens = result.tfidf_text.split()
        assert "the" in tokens
        assert "and" in tokens
        assert "on" in tokens

    def test_lemmatization_normalizes_be_verbs(self, prep):
        """
        spaCy lemmatizes 'is/are/was/were' to 'be', collapsing 4+ surface
        forms into one token without losing the concept.
        """
        result = prep.process_one("the server is down and was failing")
        tokens = result.tfidf_text.split()
        assert "be" in tokens
        assert "is" not in tokens
        assert "was" not in tokens

    def test_state_words_preserved(self, prep):
        """IT-meaningful words like 'down', 'up', 'back' must survive."""
        result = prep.process_one("Server went down then came back up")
        tokens = result.tfidf_text.split()
        assert "down" in tokens
        assert "back" in tokens
        assert "up" in tokens

    def test_only_stopwords_still_produces_tokens(self, prep):
        """When input is all stopwords, we keep them rather than emptying."""
        result = prep.process_one("the and or but")
        assert result.token_count > 0


class TestStopwordsOptIn:
    """When explicitly enabled, stopword filtering should still work."""

    def test_stopwords_removed_when_enabled(self, prep_with_stopword_removal):
        result = prep_with_stopword_removal.process_one("the laptop is on the desk")
        tokens = result.tfidf_text.split()
        assert "the" not in tokens
        assert "is" not in tokens
        assert "laptop" in tokens
        assert "desk" in tokens

    def test_keep_domain_terms_protects_vocabulary(self):
        """When user supplies keep_domain_terms, those words survive removal."""
        cfg = PreprocessConfig(
            remove_stopwords=True,
            keep_domain_terms=frozenset({"down", "back"}),
        )
        prep = TicketPreprocessor(cfg)
        result = prep.process_one("the server is down and not coming back")
        tokens = result.tfidf_text.split()
        assert "the" not in tokens
        assert "is" not in tokens
        assert "down" in tokens
        assert "back" in tokens


class TestEdgeCases:
    @pytest.mark.parametrize("bad_input", [None, "", "   ", np.nan])
    def test_empty_or_null_input(self, prep, bad_input):
        result = prep.process_one(bad_input)
        assert isinstance(result, PreprocessedTicket)
        assert result.tfidf_text == ""
        assert result.token_count == 0

    def test_only_punctuation(self, prep):
        result = prep.process_one("!!! ??? ...")
        assert result.tfidf_text == ""
        assert result.token_count == 0

    def test_unicode_handled(self, prep):
        result = prep.process_one("Café résumé naïve — VPN down")
        assert "vpn" in result.tfidf_text
        assert "down" in result.tfidf_text

    def test_very_long_text(self, prep):
        long_text = "VPN is down. " * 500
        result = prep.process_one(long_text)
        assert result.token_count > 0


class TestDeterminism:
    def test_same_input_same_output(self, prep):
        text = "User U12345 reports VPN failure with error 0x80070002"
        r1 = prep.process_one(text)
        r2 = prep.process_one(text)
        assert r1 == r2

    def test_batch_matches_one_at_a_time(self, prep):
        texts = [
            "Outlook crashes on launch",
            "VPN disconnects every 5 minutes",
            "Locked out of AD account",
            "Laptop battery not charging",
        ]
        single = [prep.process_one(t) for t in texts]
        batched = prep.process_batch(texts)
        assert single == batched


class TestDataFrameAPI:
    @pytest.fixture
    def sample_df(self) -> pd.DataFrame:
        return pd.DataFrame({
            "ticket_id": ["INC1", "INC2", "INC3"],
            "description": [
                "VPN down for user@corp.com",
                "Outlook crashing repeatedly",
                "Need access to Snowflake",
            ],
            "category": ["Network", "Software", "Access"],
        })

    def test_returns_new_dataframe_does_not_mutate(self, prep, sample_df):
        original_cols = set(sample_df.columns)
        out = prep.process_dataframe(sample_df, text_col="description")
        assert set(sample_df.columns) == original_cols
        assert "raw_text" in out.columns
        assert "tfidf_text" in out.columns
        assert "token_count" in out.columns

    def test_row_count_preserved(self, prep, sample_df):
        out = prep.process_dataframe(sample_df, text_col="description")
        assert len(out) == len(sample_df)

    def test_existing_columns_preserved(self, prep, sample_df):
        out = prep.process_dataframe(sample_df, text_col="description")
        assert "ticket_id" in out.columns
        assert "category" in out.columns
        assert (out["ticket_id"] == sample_df["ticket_id"]).all()

    def test_missing_column_raises(self, prep, sample_df):
        with pytest.raises(KeyError, match="not in DataFrame"):
            prep.process_dataframe(sample_df, text_col="nonexistent_col")

    def test_handles_nan_descriptions(self, prep):
        df = pd.DataFrame({"description": ["VPN down", None, np.nan, ""]})
        out = prep.process_dataframe(df, text_col="description")
        assert len(out) == 4
        assert out.loc[1, "tfidf_text"] == ""
        assert out.loc[2, "tfidf_text"] == ""
        assert out.loc[3, "tfidf_text"] == ""


class TestPatternRegistry:
    def test_all_pii_patterns_have_replacements(self):
        for pattern, replacement in PII_PATTERNS:
            assert pattern is not None
            assert replacement.strip().startswith("<")
            assert replacement.strip().endswith(">")