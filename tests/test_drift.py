"""
Tests for the drift monitoring module.

Verifies:
  - Reference/current window split is correct
  - Reports generate without crashing
  - Output HTML files are non-empty
  - Predictions get added correctly
  - Window-too-small protection fires

Skip-cleanly when prerequisites missing.

Run:
    pytest tests/test_drift.py -v
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.classifier import TicketClassifier
from src.config import TICKETS_CSV


def _prereqs_met() -> bool:
    return TICKETS_CSV.exists() and TicketClassifier.DEFAULT_MODEL_PATH.exists()


pytestmark = pytest.mark.skipif(
    not _prereqs_met(),
    reason="Drift tests need data + trained classifier. Run "
           "'.\\make.ps1 data' and '.\\make.ps1 train-quick' first."
)


class TestWindowSplit:
    def test_split_partitions_data(self):
        from monitoring.drift_report import load_tickets, split_reference_current
        df = load_tickets()
        ref, cur = split_reference_current(df, current_days=30)
        assert len(ref) + len(cur) == len(df)

    def test_current_is_most_recent(self):
        from monitoring.drift_report import load_tickets, split_reference_current
        df = load_tickets()
        ref, cur = split_reference_current(df, current_days=30)
        assert cur["created_at"].min() > ref["created_at"].max()

    def test_reference_days_window(self):
        """Bounded reference window: only the N days before current."""
        from monitoring.drift_report import load_tickets, split_reference_current
        df = load_tickets()
        ref, cur = split_reference_current(df, current_days=30, reference_days=60)
        if len(ref) > 0:
            ref_span_days = (ref["created_at"].max() -
                             ref["created_at"].min()).days
            assert ref_span_days <= 60


class TestAddPredictions:
    def test_predictions_added(self):
        from monitoring.drift_report import add_predictions, load_tickets
        df = load_tickets().head(50)
        out = add_predictions(df)
        assert "predicted_category" in out.columns
        assert "predicted_priority" in out.columns
        assert len(out) == len(df)

    def test_predicted_categories_in_valid_set(self):
        from monitoring.drift_report import add_predictions, load_tickets
        from src.config import CATEGORIES
        df = load_tickets().head(50)
        out = add_predictions(df)
        assert set(out["predicted_category"]).issubset(set(CATEGORIES))


class TestReportGeneration:
    def test_data_drift_report_creates_html(self, tmp_path):
        from monitoring.drift_report import (
            generate_data_drift_report,
            load_tickets,
            split_reference_current,
        )
        df = load_tickets()
        ref, cur = split_reference_current(df, current_days=30)
        out = tmp_path / "drift.html"
        generate_data_drift_report(ref, cur, out)
        assert out.exists()
        assert out.stat().st_size > 1000  # real HTML, not empty

    def test_target_drift_report_creates_html(self, tmp_path):
        from monitoring.drift_report import (
            generate_target_drift_report,
            load_tickets,
            split_reference_current,
        )
        df = load_tickets()
        ref, cur = split_reference_current(df, current_days=30)
        out = tmp_path / "target_drift.html"
        generate_target_drift_report(ref, cur, out, target="category")
        assert out.exists()
        assert out.stat().st_size > 1000


class TestWindowSizeGuards:
    def test_too_small_window_raises(self, tmp_path, monkeypatch):
        """Drift tests are unreliable on tiny samples; should refuse to run."""
        from monitoring.drift_report import run_full_drift_suite

        # 0 days -> empty current window -> should raise
        with pytest.raises(ValueError, match="too small"):
            run_full_drift_suite(current_days=0, output_dir=tmp_path)
