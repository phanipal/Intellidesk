"""
Ticket classifier for category and priority prediction.

Separate TF-IDF + XGBoost pipelines for each target. Prediction applies the
same preprocessing used during training to avoid train/serve skew.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

from src.config import CATEGORIES, MODELS_DIR, PRIORITIES

logger = logging.getLogger("intellidesk.classifier")


TFIDF_PARAMS: Dict = {
    "max_df": 0.95,
    "min_df": 2,
    "ngram_range": (1, 2),
    "max_features": 20000,
    "sublinear_tf": True,
}

XGB_PARAMS: Dict = {
    "n_estimators": 300,
    "max_depth": 6,
    "learning_rate": 0.1,
    "objective": "multi:softprob",
    "eval_metric": "mlogloss",
    "n_jobs": -1,
    "tree_method": "hist",
    "random_state": 42,
}


@dataclass(frozen=True)
class Prediction:
    """One prediction result."""
    category: str
    category_confidence: float
    priority: str
    priority_confidence: float

    def to_dict(self) -> Dict[str, float | str]:
        return asdict(self)


@dataclass
class TrainingMetrics:
    """Metrics from a single training run for one target."""
    target: str
    macro_f1: float
    weighted_f1: float
    accuracy: float
    per_class_report: str
    confusion_matrix: List[List[int]]
    class_labels: List[str]
    n_train: int
    n_test: int


class TicketClassifier:
    """
    Dual-output classifier wrapping two sklearn Pipelines, one per target.

    predict() and predict_batch() apply the SAME preprocessing as training
    to eliminate train/serve skew. Callers pass raw ticket text.
    """

    DEFAULT_MODEL_PATH = MODELS_DIR / "classifier.joblib"

    def __init__(self):
        self.cat_pipeline: Optional[Pipeline] = None
        self.cat_encoder: Optional[LabelEncoder] = None
        self.pri_pipeline: Optional[Pipeline] = None
        self.pri_encoder: Optional[LabelEncoder] = None
        self.is_trained: bool = False
        self._preprocessor = None

    @staticmethod
    def _build_pipeline(num_classes: int) -> Pipeline:
        """Build a fresh TF-IDF + XGBoost pipeline for one target."""
        return Pipeline([
            ("tfidf", TfidfVectorizer(**TFIDF_PARAMS)),
            ("xgb", XGBClassifier(num_class=num_classes, **XGB_PARAMS)),
        ])

    def _preprocess_one(self, text: str) -> str:
        """
        Apply the same preprocessing used during training.
        Lazy-loads spaCy on first call.
        """
        if self._preprocessor is None:
            from src.preprocess import TicketPreprocessor
            self._preprocessor = TicketPreprocessor()
        return self._preprocessor.process_one(text or "").tfidf_text

    def train(
        self,
        df: pd.DataFrame,
        text_col: str = "tfidf_text",
        category_col: str = "category",
        priority_col: str = "priority",
        test_size: float = 0.2,
        random_state: int = 42,
        log_to_mlflow: bool = True,
    ) -> Dict[str, TrainingMetrics]:
        """Train both classifiers and return metrics per target."""
        self._validate_training_df(df, text_col, category_col, priority_col)

        train_df, test_df = train_test_split(
            df,
            test_size=test_size,
            random_state=random_state,
            stratify=df[category_col],
        )
        logger.info("Train/test split: %d / %d rows", len(train_df), len(test_df))

        results: Dict[str, TrainingMetrics] = {}

        logger.info("Training category classifier...")
        results["category"] = self._train_one_target(
            train_df, test_df,
            text_col=text_col, label_col=category_col,
            target_name="category", class_order=list(CATEGORIES),
        )

        logger.info("Training priority classifier...")
        results["priority"] = self._train_one_target(
            train_df, test_df,
            text_col=text_col, label_col=priority_col,
            target_name="priority", class_order=list(PRIORITIES),
        )

        self.is_trained = True

        if log_to_mlflow:
            self._log_to_mlflow(results, train_df, test_df)

        return results

    def _train_one_target(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        text_col: str,
        label_col: str,
        target_name: str,
        class_order: List[str],
    ) -> TrainingMetrics:
        """Train one classifier (category OR priority). Returns metrics."""
        encoder = LabelEncoder().fit(class_order)
        y_train = encoder.transform(train_df[label_col])
        y_test = encoder.transform(test_df[label_col])

        pipeline = self._build_pipeline(num_classes=len(class_order))

        # Category (~3:1) does not need weighting. Priority (~12:1) gets
        # sqrt-scaled balanced weights, gentler than full inverse-frequency
        # so we don't over-correct.
        fit_kwargs = {}
        if target_name == "priority":
            balanced = compute_sample_weight(class_weight="balanced", y=y_train)
            fit_kwargs["xgb__sample_weight"] = balanced ** 0.5

        pipeline.fit(train_df[text_col].fillna(""), y_train, **fit_kwargs)

        y_pred = pipeline.predict(test_df[text_col].fillna(""))

        macro_f1 = f1_score(y_test, y_pred, average="macro")
        weighted_f1 = f1_score(y_test, y_pred, average="weighted")
        accuracy = (y_pred == y_test).mean()

        # LabelEncoder sorts classes alphabetically. Use the encoder's actual
        # integer mapping so target_names align with the underlying class
        # indices in the report and confusion matrix.
        ordered_labels = encoder.transform(class_order)
        report = classification_report(
            y_test, y_pred,
            labels=ordered_labels,
            target_names=class_order,
            zero_division=0,
        )
        cm = confusion_matrix(y_test, y_pred, labels=ordered_labels)

        logger.info(
            "%s: macro_f1=%.3f weighted_f1=%.3f accuracy=%.3f",
            target_name, macro_f1, weighted_f1, accuracy,
        )

        if target_name == "category":
            self.cat_pipeline = pipeline
            self.cat_encoder = encoder
        else:
            self.pri_pipeline = pipeline
            self.pri_encoder = encoder

        return TrainingMetrics(
            target=target_name,
            macro_f1=float(macro_f1),
            weighted_f1=float(weighted_f1),
            accuracy=float(accuracy),
            per_class_report=report,
            confusion_matrix=cm.tolist(),
            class_labels=class_order,
            n_train=len(train_df),
            n_test=len(test_df),
        )

    @staticmethod
    def _validate_training_df(df: pd.DataFrame, text_col: str,
                              cat_col: str, pri_col: str) -> None:
        for col in (text_col, cat_col, pri_col):
            if col not in df.columns:
                raise KeyError(f"Required column '{col}' missing from DataFrame")

        if df[cat_col].isna().any():
            raise ValueError(f"Found NaN in {cat_col}")
        if df[pri_col].isna().any():
            raise ValueError(f"Found NaN in {pri_col}")

        unknown_cats = set(df[cat_col].unique()) - set(CATEGORIES)
        if unknown_cats:
            raise ValueError(f"Unknown categories: {unknown_cats}. "
                             f"Expected: {CATEGORIES}")
        unknown_pris = set(df[pri_col].unique()) - set(PRIORITIES)
        if unknown_pris:
            raise ValueError(f"Unknown priorities: {unknown_pris}. "
                             f"Expected: {PRIORITIES}")

    def predict(self, text: str) -> Prediction:
        """
        Predict category and priority for a single RAW ticket description.
        Preprocessing is applied internally; callers pass raw text.
        """
        if not self.is_trained:
            raise RuntimeError("Classifier not trained. Call train() or load() first.")

        cleaned = self._preprocess_one(text)

        cat_proba = self.cat_pipeline.predict_proba([cleaned])[0]
        cat_idx = int(np.argmax(cat_proba))
        cat_label = self.cat_encoder.inverse_transform([cat_idx])[0]

        pri_proba = self.pri_pipeline.predict_proba([cleaned])[0]
        pri_idx = int(np.argmax(pri_proba))
        pri_label = self.pri_encoder.inverse_transform([pri_idx])[0]

        return Prediction(
            category=str(cat_label),
            category_confidence=float(cat_proba[cat_idx]),
            priority=str(pri_label),
            priority_confidence=float(pri_proba[pri_idx]),
        )

    def predict_batch(self, texts: List[str]) -> List[Prediction]:
        """Batch prediction for raw ticket descriptions."""
        if not self.is_trained:
            raise RuntimeError("Classifier not trained. Call train() or load() first.")

        cleaned_texts = [self._preprocess_one(t) for t in texts]

        cat_proba = self.cat_pipeline.predict_proba(cleaned_texts)
        cat_idxs = cat_proba.argmax(axis=1)
        cat_labels = self.cat_encoder.inverse_transform(cat_idxs)

        pri_proba = self.pri_pipeline.predict_proba(cleaned_texts)
        pri_idxs = pri_proba.argmax(axis=1)
        pri_labels = self.pri_encoder.inverse_transform(pri_idxs)

        return [
            Prediction(
                category=str(cat_labels[i]),
                category_confidence=float(cat_proba[i, cat_idxs[i]]),
                priority=str(pri_labels[i]),
                priority_confidence=float(pri_proba[i, pri_idxs[i]]),
            )
            for i in range(len(texts))
        ]

    def save(self, path: Optional[Path] = None) -> Path:
        """Save the trained classifier (both pipelines + encoders) to disk."""
        if not self.is_trained:
            raise RuntimeError("Cannot save an untrained classifier.")

        path = path or self.DEFAULT_MODEL_PATH
        path.parent.mkdir(parents=True, exist_ok=True)

        joblib.dump({
            "cat_pipeline": self.cat_pipeline,
            "cat_encoder": self.cat_encoder,
            "pri_pipeline": self.pri_pipeline,
            "pri_encoder": self.pri_encoder,
        }, path)

        logger.info("Saved classifier -> %s", path)
        return path

    @classmethod
    def load(cls, path: Optional[Path] = None) -> "TicketClassifier":
        """Load a trained classifier from disk."""
        path = path or cls.DEFAULT_MODEL_PATH
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")

        bundle = joblib.load(path)
        clf = cls()
        clf.cat_pipeline = bundle["cat_pipeline"]
        clf.cat_encoder = bundle["cat_encoder"]
        clf.pri_pipeline = bundle["pri_pipeline"]
        clf.pri_encoder = bundle["pri_encoder"]
        clf.is_trained = True
        logger.info("Loaded classifier <- %s", path)
        return clf

    def _log_to_mlflow(
        self,
        metrics: Dict[str, TrainingMetrics],
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
    ) -> None:
        """Log this training run to MLflow. Failures are non-fatal."""
        try:
            import mlflow
            import mlflow.sklearn
        except ImportError:
            logger.warning("MLflow not installed - skipping experiment logging")
            return

        try:
            mlflow.set_experiment("intellidesk-classifier")
            with mlflow.start_run():
                for k, v in TFIDF_PARAMS.items():
                    mlflow.log_param(f"tfidf_{k}", v)
                for k, v in XGB_PARAMS.items():
                    mlflow.log_param(f"xgb_{k}", v)
                mlflow.log_param("n_train", len(train_df))
                mlflow.log_param("n_test", len(test_df))

                for target, m in metrics.items():
                    mlflow.log_metric(f"{target}_macro_f1", m.macro_f1)
                    mlflow.log_metric(f"{target}_weighted_f1", m.weighted_f1)
                    mlflow.log_metric(f"{target}_accuracy", m.accuracy)

                for target, m in metrics.items():
                    report_path = MODELS_DIR / f"_mlflow_{target}_report.txt"
                    cm_path = MODELS_DIR / f"_mlflow_{target}_cm.json"
                    report_path.parent.mkdir(parents=True, exist_ok=True)
                    report_path.write_text(m.per_class_report)
                    cm_path.write_text(json.dumps({
                        "labels": m.class_labels,
                        "matrix": m.confusion_matrix,
                    }, indent=2))
                    mlflow.log_artifact(str(report_path))
                    mlflow.log_artifact(str(cm_path))
                    report_path.unlink()
                    cm_path.unlink()

                mlflow.sklearn.log_model(self.cat_pipeline, "category_pipeline")
                mlflow.sklearn.log_model(self.pri_pipeline, "priority_pipeline")

                logger.info("MLflow run logged successfully")
        except Exception as exc:
            logger.warning("MLflow logging failed: %s (training succeeded)", exc)


def main() -> None:
    """CLI: preprocess data/tickets.csv, train both classifiers, save model."""
    import argparse

    from src.config import TICKETS_CSV
    from src.preprocess import TicketPreprocessor

    parser = argparse.ArgumentParser(description="Train ticket classifier.")
    parser.add_argument("--data", type=str, default=str(TICKETS_CSV))
    parser.add_argument("--no-mlflow", action="store_true",
                        help="Skip MLflow logging")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    logger.info("Loading data from %s", args.data)
    df = pd.read_csv(args.data)
    logger.info("Loaded %d rows", len(df))

    logger.info("Preprocessing text...")
    prep = TicketPreprocessor()
    df = prep.process_dataframe(df, text_col="description")

    clf = TicketClassifier()
    metrics = clf.train(df, log_to_mlflow=not args.no_mlflow)

    print("\n" + "=" * 70)
    print("  TRAINING COMPLETE")
    print("=" * 70)
    for target, m in metrics.items():
        print(f"\n  {target.upper()}")
        print(f"    macro_f1:    {m.macro_f1:.3f}")
        print(f"    weighted_f1: {m.weighted_f1:.3f}")
        print(f"    accuracy:    {m.accuracy:.3f}")
        print(f"\n  Per-class report:\n{m.per_class_report}")

    clf.save()
    print("=" * 70)


if __name__ == "__main__":
    main()
