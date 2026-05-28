"""
Drift monitoring for IntelliDesk via Evidently AI.

Generates HTML reports comparing current and reference ticket windows for data drift,
target drift, and optional classification quality metrics.
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd

from src.config import REPORTS_DIR, TICKETS_CSV

logger = logging.getLogger("intellidesk.monitoring")


@dataclass(frozen=True)
class DriftConfig:
    """Knobs for what's compared and how."""
    # Default windows: 30 days "current" vs everything earlier as "reference"
    current_days: int = 30
    # drift tests are unreliable on tiny samples
    min_window_rows: int = 100
    numeric_features: tuple = (
        "resolution_hours",
        "sla_target_hours",
    )
    categorical_features: tuple = (
        "category",
        "priority",
        "subcategory",
        "status",
        "assigned_group",
    )


DEFAULT_CONFIG = DriftConfig()


def load_tickets() -> pd.DataFrame:
    """Load tickets and compute helper columns drift reports use."""
    if not TICKETS_CSV.exists():
        raise FileNotFoundError(
            f"Tickets file not found: {TICKETS_CSV}. "
            f"Run '.\\make.ps1 data' first."
        )

    df = pd.read_csv(
        TICKETS_CSV,
        parse_dates=["created_at", "first_response_at", "resolved_at"],
    )
    # Description length shifts when noise patterns or the ticket-intake
    # UI changes, which makes it a strong drift indicator.
    df["description_length"] = df["description"].fillna("").str.len()
    return df


def split_reference_current(
    df: pd.DataFrame,
    current_days: int = 30,
    reference_days: Optional[int] = None,
) -> tuple:
    """
    Split tickets into reference vs current windows by created_at.

    Args:
        df: Full ticket DataFrame.
        current_days: How many days back from max date define the "current" window.
        reference_days: If provided, reference is the N days BEFORE the current
                        window. If None, reference is everything before current.

    Returns:
        (reference_df, current_df) tuple.
    """
    max_date = df["created_at"].max()
    current_start = max_date - pd.Timedelta(days=current_days)

    current = df[df["created_at"] > current_start].copy()

    if reference_days is None:
        reference = df[df["created_at"] <= current_start].copy()
    else:
        ref_start = current_start - pd.Timedelta(days=reference_days)
        reference = df[
            (df["created_at"] > ref_start) & (df["created_at"] <= current_start)
        ].copy()

    logger.info(
        "Reference window: %s → %s (%d rows)",
        reference["created_at"].min() if len(reference) else "n/a",
        reference["created_at"].max() if len(reference) else "n/a",
        len(reference),
    )
    logger.info(
        "Current window:   %s → %s (%d rows)",
        current["created_at"].min() if len(current) else "n/a",
        current["created_at"].max() if len(current) else "n/a",
        len(current),
    )
    return reference, current


def _build_column_mapping(config: DriftConfig):
    """
    Tell Evidently which columns are which type. Required for correct
    drift tests (KS for numeric, chi-square for categorical).
    """
    from evidently import ColumnMapping

    return ColumnMapping(
        numerical_features=list(config.numeric_features) + ["description_length"],
        categorical_features=list(config.categorical_features),
        text_features=["description"],
    )


def generate_data_drift_report(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    output_path: Path,
    config: DriftConfig = DEFAULT_CONFIG,
) -> Path:
    """
    Data drift report: distribution shifts across all monitored features.

    Generates an HTML report with per-feature statistical tests, drift scores,
    and visualizations (histograms, distribution overlays, correlation heatmap).
    """
    from evidently.report import Report
    from evidently.metric_preset import DataDriftPreset, DataQualityPreset

    column_mapping = _build_column_mapping(config)

    report = Report(metrics=[
        DataDriftPreset(),
        DataQualityPreset(),
    ])
    report.run(
        reference_data=reference,
        current_data=current,
        column_mapping=column_mapping,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    report.save_html(str(output_path))
    logger.info("Data drift report -> %s", output_path)
    return output_path


def generate_target_drift_report(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    output_path: Path,
    target: str = "category",
) -> Path:
    """
    Target drift report: tracks shifts in our prediction target.

    Use one report per target (category, priority). Surfaces the question:
    "Has the LABEL we predict shifted, regardless of feature shifts?"
    Important for SLA dashboards that assume P1 stays around 4%.
    """
    from evidently.report import Report
    from evidently.metric_preset import TargetDriftPreset
    from evidently import ColumnMapping

    column_mapping = ColumnMapping(target=target)

    report = Report(metrics=[TargetDriftPreset()])
    report.run(
        reference_data=reference,
        current_data=current,
        column_mapping=column_mapping,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    report.save_html(str(output_path))
    logger.info("Target drift report (%s) -> %s", target, output_path)
    return output_path


def generate_classification_quality_report(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    output_path: Path,
    target: str = "category",
) -> Path:
    """
    Classification quality report: actual vs predicted comparison.

    Requires that current/reference DataFrames have BOTH:
      - 'category' or 'priority' (ground truth)
      - 'predicted_category' or 'predicted_priority' (model output)

    The CLI populates these by running the classifier on the windows. In
    production you'd already have predictions logged from /triage calls.
    """
    from evidently.report import Report
    from evidently.metric_preset import ClassificationPreset
    from evidently import ColumnMapping

    pred_col = f"predicted_{target}"
    column_mapping = ColumnMapping(target=target, prediction=pred_col)

    report = Report(metrics=[ClassificationPreset()])
    report.run(
        reference_data=reference,
        current_data=current,
        column_mapping=column_mapping,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    report.save_html(str(output_path))
    logger.info("Classification quality report (%s) -> %s", target, output_path)
    return output_path


def add_predictions(df: pd.DataFrame) -> pd.DataFrame:
    """Run classifier on a DataFrame and add predicted_category/predicted_priority."""
    from src.classifier import TicketClassifier

    if not TicketClassifier.DEFAULT_MODEL_PATH.exists():
        raise FileNotFoundError(
            "Classifier not trained. Run '.\\make.ps1 train-quick' first."
        )

    clf = TicketClassifier.load()
    logger.info("Running classifier on %d tickets...", len(df))
    predictions = clf.predict_batch(df["description"].fillna("").tolist())
    df = df.copy()
    df["predicted_category"] = [p.category for p in predictions]
    df["predicted_priority"] = [p.priority for p in predictions]
    return df


@dataclass
class DriftRunSummary:
    """Quick scan of which reports were generated and where."""
    reference_rows: int
    current_rows: int
    reports_generated: dict


def run_full_drift_suite(
    current_days: int = 30,
    reference_days: Optional[int] = None,
    include_quality: bool = False,
    output_dir: Path = REPORTS_DIR,
) -> DriftRunSummary:
    """
    Generate the full drift report bundle.

    Args:
        current_days: Window size for current data (most recent N days).
        reference_days: Window size for reference. None = everything before current.
        include_quality: Also generate a classification quality report. Slower
                         since it runs the classifier on both windows.
        output_dir: Where to save reports.
    """
    df = load_tickets()
    reference, current = split_reference_current(df, current_days, reference_days)

    if len(reference) < DEFAULT_CONFIG.min_window_rows:
        raise ValueError(
            f"Reference window too small ({len(reference)} rows < "
            f"{DEFAULT_CONFIG.min_window_rows}). Use a larger reference window."
        )
    if len(current) < DEFAULT_CONFIG.min_window_rows:
        raise ValueError(
            f"Current window too small ({len(current)} rows < "
            f"{DEFAULT_CONFIG.min_window_rows}). Use a longer current_days "
            f"or wait until more tickets have been logged."
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    reports = {}

    path = output_dir / f"drift_data_{timestamp}.html"
    generate_data_drift_report(reference, current, path)
    reports["data_drift"] = path

    # one report per prediction target
    for target in ("category", "priority"):
        path = output_dir / f"drift_target_{target}_{timestamp}.html"
        generate_target_drift_report(reference, current, path, target=target)
        reports[f"target_drift_{target}"] = path

    if include_quality:
        ref_with_pred = add_predictions(reference)
        cur_with_pred = add_predictions(current)
        for target in ("category", "priority"):
            path = output_dir / f"drift_quality_{target}_{timestamp}.html"
            generate_classification_quality_report(
                ref_with_pred, cur_with_pred, path, target=target
            )
            reports[f"quality_{target}"] = path

    # Copy each as "latest" for the dashboard to pick up
    for name, path in list(reports.items()):
        latest_path = output_dir / f"{name.replace('drift_', '')}_latest.html"
        latest_path.write_bytes(path.read_bytes())
        reports[f"{name}_latest"] = latest_path

    return DriftRunSummary(
        reference_rows=len(reference),
        current_rows=len(current),
        reports_generated=reports,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate drift monitoring reports for IntelliDesk."
    )
    parser.add_argument("--current-days", type=int, default=30,
                        help="Days back from max date that define current window")
    parser.add_argument("--reference-days", type=int, default=None,
                        help="Reference window size (default: everything before current)")
    parser.add_argument("--include-quality", action="store_true",
                        help="Also generate classification quality reports (slower)")
    parser.add_argument("--output-dir", type=Path, default=REPORTS_DIR)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    summary = run_full_drift_suite(
        current_days=args.current_days,
        reference_days=args.reference_days,
        include_quality=args.include_quality,
        output_dir=args.output_dir,
    )

    print("\n" + "=" * 70)
    print("  DRIFT REPORT BUNDLE GENERATED")
    print("=" * 70)
    print(f"  Reference window: {summary.reference_rows:,} tickets")
    print(f"  Current window:   {summary.current_rows:,} tickets")
    print(f"\n  Reports:")
    for name, path in summary.reports_generated.items():
        if not name.endswith("_latest"):
            print(f"    {name:30s} -> {path}")
    print("=" * 70)
    print(f"\n  Open the HTML files in your browser to view full reports.")
    print(f"  'data_drift_latest.html' is updated each run for the dashboard.\n")


if __name__ == "__main__":
    main()
