"""
Unit tests for the synthetic ticket generator.
All tests run in-memory; no files are written.

Run:
    pytest tests/test_generate_data.py -v
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.generate_data import (
    TEMPLATES,
    _resolve_sla_targets,
    generate_tickets,
)

EXPECTED_COLUMNS = {
    "ticket_id", "created_at", "first_response_at", "resolved_at",
    "status", "category", "subcategory", "priority", "description",
    "requester_id", "assigned_group", "kb_article_id",
    "sla_target_hours", "resolution_hours", "sla_met",
}

VALID_CATEGORIES = {"Network", "Software", "Hardware", "Access"}
VALID_PRIORITIES = {"P1", "P2", "P3", "P4"}
VALID_STATUSES = {"Resolved", "In Progress", "Open"}
OUTAGE_SUBCATS = {"Site Outage", "Production App Down", "Critical Access Outage"}


@pytest.fixture(scope="module")
def df() -> pd.DataFrame:
    """Generate once, reuse across tests for speed."""
    return generate_tickets(n_tickets=2000, seed=42)


def test_row_count(df):
    assert len(df) == 2000


def test_columns_match_contract(df):
    assert set(df.columns) == EXPECTED_COLUMNS


def test_ticket_ids_unique(df):
    assert df["ticket_id"].is_unique
    assert df["ticket_id"].str.startswith("INC").all()


def test_categories_in_allowed_set(df):
    assert set(df["category"].unique()).issubset(VALID_CATEGORIES)


def test_priorities_in_allowed_set(df):
    assert set(df["priority"].unique()).issubset(VALID_PRIORITIES)


def test_statuses_in_allowed_set(df):
    assert set(df["status"].unique()).issubset(VALID_STATUSES)


def test_descriptions_non_empty_and_no_unfilled_tokens(df):
    assert df["description"].str.len().min() > 5
    assert not df["description"].str.contains(r"\{\w+\}", regex=True).any()


def test_resolved_tickets_have_resolution_fields(df):
    resolved = df[df["status"] == "Resolved"]
    assert resolved["resolved_at"].notna().all()
    assert resolved["resolution_hours"].notna().all()
    assert resolved["sla_met"].notna().all()
    assert resolved["first_response_at"].notna().all()


def test_open_tickets_have_no_response_or_resolution(df):
    open_tix = df[df["status"] == "Open"]
    assert open_tix["resolved_at"].isna().all()
    assert open_tix["first_response_at"].isna().all()
    assert open_tix["resolution_hours"].isna().all()


def test_in_progress_tickets_have_response_no_resolution(df):
    in_prog = df[df["status"] == "In Progress"]
    assert in_prog["first_response_at"].notna().all()
    assert in_prog["resolved_at"].isna().all()


def test_resolved_at_after_created_at(df):
    resolved = df[df["status"] == "Resolved"]
    assert (resolved["resolved_at"] > resolved["created_at"]).all()


def test_first_response_after_created(df):
    responded = df[df["first_response_at"].notna()]
    assert (responded["first_response_at"] >= responded["created_at"]).all()


def test_sla_met_consistent_with_resolution_hours(df):
    resolved = df[df["status"] == "Resolved"]
    expected = resolved["resolution_hours"] <= resolved["sla_target_hours"]
    assert (resolved["sla_met"] == expected).all()


def test_sla_targets_match_priority(df):
    sla_map = _resolve_sla_targets()
    for priority, target in sla_map.items():
        subset = df[df["priority"] == priority]
        assert (subset["sla_target_hours"] == target).all()


def test_category_mix_realistic(df):
    mix = df["category"].value_counts(normalize=True)
    assert 0.34 <= mix["Software"] <= 0.46
    assert 0.22 <= mix["Access"] <= 0.34
    assert 0.13 <= mix["Hardware"] <= 0.23
    assert 0.09 <= mix["Network"] <= 0.19


def test_priority_long_tail(df):
    """P3/P4 dominant, P1 rare (realistic for enterprise IT)."""
    mix = df["priority"].value_counts(normalize=True)
    assert mix["P3"] >= mix["P1"], "P3 should dominate over P1"
    assert mix["P1"] < 0.10, "P1 should be rare (typically 2-6%)"


def test_p1_concentrated_in_outage_subcategories(df):
    """
    Within outage subcategories, P1 rate should be the majority and far
    higher than in non-outage subcategories. This is the core class-imbalance
    pattern the classifier needs to learn.
    """
    outage_tix = df[df["subcategory"].isin(OUTAGE_SUBCATS)]
    non_outage_tix = df[~df["subcategory"].isin(OUTAGE_SUBCATS)]

    if len(outage_tix) == 0:
        pytest.skip("No outage tickets in this sample")

    outage_p1_rate = (outage_tix["priority"] == "P1").mean()
    non_outage_p1_rate = (non_outage_tix["priority"] == "P1").mean()

    assert outage_p1_rate > 0.5, (
        f"Outage P1 rate is {outage_p1_rate:.2%}, expected majority"
    )
    assert outage_p1_rate > 5 * non_outage_p1_rate, (
        f"Outage P1 rate {outage_p1_rate:.2%} should be >> "
        f"non-outage P1 rate {non_outage_p1_rate:.2%}"
    )


def test_seed_reproducibility():
    """Same seed → byte-identical DataFrames."""
    df1 = generate_tickets(n_tickets=500, seed=123)
    df2 = generate_tickets(n_tickets=500, seed=123)
    pd.testing.assert_frame_equal(df1, df2)


def test_different_seeds_produce_different_data():
    df1 = generate_tickets(n_tickets=500, seed=1)
    df2 = generate_tickets(n_tickets=500, seed=2)
    assert not df1["description"].equals(df2["description"])


def test_every_template_referenced(df):
    """
    Across 2000 rows, most templates should appear. Outage templates have
    very low weight (0.05) so up to 2 may be absent in a small sample.
    """
    used_kbs = set(df["kb_article_id"].unique())
    template_kbs = {t.resolution_kb_id for t in TEMPLATES}
    missing = template_kbs - used_kbs
    assert len(missing) <= 2, f"Too many templates never sampled: {missing}"


class TestNoiseInjection:
    def test_noise_disabled_produces_shorter_descriptions_on_average(self):
        """With noise=0, descriptions are pure templates with no appended fragments."""
        df_no_noise = generate_tickets(n_tickets=500, seed=42, noise_rate=0.0)
        df_with_noise = generate_tickets(n_tickets=500, seed=42, noise_rate=0.5)
        assert (df_with_noise["description"].str.len().mean() >
                df_no_noise["description"].str.len().mean())

    def test_noise_does_not_change_label_distribution(self):
        """Ground-truth category mix stays consistent regardless of noise."""
        df_no_noise = generate_tickets(n_tickets=2000, seed=42, noise_rate=0.0)
        df_with_noise = generate_tickets(n_tickets=2000, seed=42, noise_rate=0.5)
        no_noise_mix = df_no_noise["category"].value_counts(normalize=True).sort_index()
        noisy_mix = df_with_noise["category"].value_counts(normalize=True).sort_index()
        for cat in no_noise_mix.index:
            assert abs(no_noise_mix[cat] - noisy_mix[cat]) < 0.05

    def test_noise_default_is_moderate(self):
        """Default noise rate should be neither 0 nor extreme."""
        from src.generate_data import generate_tickets
        import inspect
        sig = inspect.signature(generate_tickets)
        default = sig.parameters["noise_rate"].default
        assert 0.05 <= default <= 0.20
