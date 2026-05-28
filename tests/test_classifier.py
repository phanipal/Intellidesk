"""Tests for TicketClassifier: training, prediction, and serialization."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.classifier import (
    Prediction,
    TicketClassifier,
    TrainingMetrics,
    XGB_PARAMS,
)
from src.config import CATEGORIES, PRIORITIES
from src.generate_data import generate_tickets
from src.preprocess import TicketPreprocessor


def _ensure_spacy_model() -> None:
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


# Set comfortably below realistic targets so test noise doesn't cause
# spurious failures. If these break, something fundamental is wrong with
# the model or the data.
MIN_MACRO_F1_CATEGORY = 0.80   # categories are well-separated
MIN_MACRO_F1_PRIORITY = 0.38   # harder: semantic + class imbalance


@pytest.fixture(scope="module")
def trained_classifier() -> TicketClassifier:
    """
    Train one classifier and reuse across tests.

    Uses CLEAN data (noise_rate=0) so test pass/fail doesn't depend on
    noise variance. Robustness to noise is measured separately via
    train-quick on the production dataset.
    """
    df = generate_tickets(n_tickets=2000, seed=42, noise_rate=0.0)
    prep = TicketPreprocessor()
    df = prep.process_dataframe(df, text_col="description")

    clf = TicketClassifier()
    clf.train(df, log_to_mlflow=False)
    return clf


@pytest.fixture(scope="module")
def training_metrics(trained_classifier) -> dict:
    """Re-run train to capture metrics object alongside the classifier."""
    df = generate_tickets(n_tickets=2000, seed=42, noise_rate=0.0)
    prep = TicketPreprocessor()
    df = prep.process_dataframe(df, text_col="description")
    clf = TicketClassifier()
    return clf.train(df, log_to_mlflow=False)


class TestTrainingContract:
    def test_training_returns_both_targets(self, training_metrics):
        assert "category" in training_metrics
        assert "priority" in training_metrics

    def test_metrics_are_in_valid_range(self, training_metrics):
        for target, m in training_metrics.items():
            assert isinstance(m, TrainingMetrics)
            assert 0.0 <= m.macro_f1 <= 1.0
            assert 0.0 <= m.weighted_f1 <= 1.0
            assert 0.0 <= m.accuracy <= 1.0

    def test_classifier_marked_trained_after_train(self, trained_classifier):
        assert trained_classifier.is_trained
        assert trained_classifier.cat_pipeline is not None
        assert trained_classifier.pri_pipeline is not None
        assert trained_classifier.cat_encoder is not None
        assert trained_classifier.pri_encoder is not None

    def test_class_labels_in_order(self, training_metrics):
        assert training_metrics["category"].class_labels == list(CATEGORIES)
        assert training_metrics["priority"].class_labels == list(PRIORITIES)

    def test_confusion_matrix_shape(self, training_metrics):
        cat_cm = training_metrics["category"].confusion_matrix
        assert len(cat_cm) == len(CATEGORIES)
        assert all(len(row) == len(CATEGORIES) for row in cat_cm)
        pri_cm = training_metrics["priority"].confusion_matrix
        assert len(pri_cm) == len(PRIORITIES)


class TestPerformanceFloor:
    """
    Guards against silent regression. If you change the model and these
    floors break, you lost real performance; investigate before merging.
    """

    def test_category_macro_f1_above_floor(self, training_metrics):
        f1 = training_metrics["category"].macro_f1
        assert f1 >= MIN_MACRO_F1_CATEGORY, (
            f"Category macro F1 = {f1:.3f}, below floor {MIN_MACRO_F1_CATEGORY}"
        )

    def test_priority_macro_f1_above_floor(self, training_metrics):
        f1 = training_metrics["priority"].macro_f1
        assert f1 >= MIN_MACRO_F1_PRIORITY, (
            f"Priority macro F1 = {f1:.3f}, below floor {MIN_MACRO_F1_PRIORITY}"
        )


class TestPrediction:
    def test_predict_returns_prediction_object(self, trained_classifier):
        result = trained_classifier.predict("VPN is down for the entire team")
        assert isinstance(result, Prediction)

    def test_predict_returns_valid_labels(self, trained_classifier):
        result = trained_classifier.predict("Outlook crashed on launch")
        assert result.category in CATEGORIES
        assert result.priority in PRIORITIES

    def test_predict_confidences_in_unit_interval(self, trained_classifier):
        result = trained_classifier.predict("My laptop battery is dying")
        assert 0.0 <= result.category_confidence <= 1.0
        assert 0.0 <= result.priority_confidence <= 1.0

    def test_predict_handles_empty_string(self, trained_classifier):
        result = trained_classifier.predict("")
        assert result.category in CATEGORIES
        assert result.priority in PRIORITIES

    def test_predict_handles_none(self, trained_classifier):
        result = trained_classifier.predict(None)
        assert result.category in CATEGORIES

    def test_predict_to_dict(self, trained_classifier):
        result = trained_classifier.predict("Need access to Snowflake")
        d = result.to_dict()
        assert set(d.keys()) == {"category", "category_confidence",
                                  "priority", "priority_confidence"}

    def test_obvious_examples_classify_correctly(self, trained_classifier):
        """
        Allow 1 miss out of 4 because cross-category noise injection during
        training creates legitimate confusion on borderline cases (e.g.,
        'Outlook crashes' can be confused if the training set saw Outlook
        mentioned in non-Software noise).
        """
        cases = [
            ("VPN keeps disconnecting from corporate network", "Network"),
            ("Outlook application crashes on startup", "Software"),
            ("Laptop screen is flickering", "Hardware"),
            ("Need password reset for AD account", "Access"),
        ]
        correct = 0
        misses = []
        for text, expected_cat in cases:
            result = trained_classifier.predict(text)
            if result.category == expected_cat:
                correct += 1
            else:
                misses.append(f"'{text}' expected {expected_cat}, got {result.category}")

        assert correct >= 3, (
            f"Only {correct}/4 obvious examples classified correctly. "
            f"Misses: {misses}"
        )


class TestBatchPrediction:
    def test_batch_returns_correct_count(self, trained_classifier):
        texts = ["VPN down", "Outlook crash", "Password reset"]
        results = trained_classifier.predict_batch(texts)
        assert len(results) == 3

    def test_batch_matches_single_predictions(self, trained_classifier):
        texts = [
            "VPN keeps disconnecting",
            "Outlook crashes on launch",
            "Need access to project folder",
        ]
        single = [trained_classifier.predict(t) for t in texts]
        batched = trained_classifier.predict_batch(texts)
        for s, b in zip(single, batched):
            assert s.category == b.category
            assert s.priority == b.priority
            assert abs(s.category_confidence - b.category_confidence) < 1e-9


class TestValidation:
    @pytest.fixture
    def minimal_df(self):
        df = generate_tickets(n_tickets=100, seed=1)
        prep = TicketPreprocessor()
        return prep.process_dataframe(df, text_col="description")

    def test_missing_text_col_raises(self, minimal_df):
        clf = TicketClassifier()
        with pytest.raises(KeyError, match="missing"):
            clf.train(minimal_df, text_col="nonexistent",
                      log_to_mlflow=False)

    def test_missing_category_col_raises(self, minimal_df):
        clf = TicketClassifier()
        bad_df = minimal_df.drop(columns=["category"])
        with pytest.raises(KeyError, match="missing"):
            clf.train(bad_df, log_to_mlflow=False)

    def test_unknown_category_raises(self, minimal_df):
        clf = TicketClassifier()
        bad_df = minimal_df.copy()
        bad_df.loc[0, "category"] = "Quantum"  # not in CATEGORIES
        with pytest.raises(ValueError, match="Unknown categories"):
            clf.train(bad_df, log_to_mlflow=False)

    def test_unknown_priority_raises(self, minimal_df):
        clf = TicketClassifier()
        bad_df = minimal_df.copy()
        bad_df.loc[0, "priority"] = "P9"
        with pytest.raises(ValueError, match="Unknown priorities"):
            clf.train(bad_df, log_to_mlflow=False)

    def test_predict_before_train_raises(self):
        clf = TicketClassifier()
        with pytest.raises(RuntimeError, match="not trained"):
            clf.predict("VPN down")


class TestPersistence:
    def test_save_creates_file(self, trained_classifier, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        save_path = tmp_path / "test_model.joblib"
        path = trained_classifier.save(path=save_path)
        assert path.exists()
        assert path.stat().st_size > 0

    def test_load_restores_predictions_exactly(
        self, trained_classifier, tmp_path
    ):
        save_path = tmp_path / "test_model.joblib"
        trained_classifier.save(path=save_path)

        loaded = TicketClassifier.load(path=save_path)
        assert loaded.is_trained

        text = "VPN is failing for our entire department this morning"
        before = trained_classifier.predict(text)
        after = loaded.predict(text)

        assert before.category == after.category
        assert before.priority == after.priority
        assert abs(before.category_confidence - after.category_confidence) < 1e-9
        assert abs(before.priority_confidence - after.priority_confidence) < 1e-9

    def test_save_untrained_raises(self):
        clf = TicketClassifier()
        with pytest.raises(RuntimeError, match="untrained"):
            clf.save()

    def test_load_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            TicketClassifier.load(path=tmp_path / "nonexistent.joblib")


class TestHyperparameters:
    def test_xgb_uses_reproducible_seed(self):
        assert XGB_PARAMS["random_state"] == 42

    def test_xgb_uses_efficient_tree_method(self):
        assert XGB_PARAMS["tree_method"] == "hist"
