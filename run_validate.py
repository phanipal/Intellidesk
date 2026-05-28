"""
Validation suite for IntelliDesk.

Runs checks across data integrity, model artifacts, classification metrics,
retrieval relevance, and end-to-end pipeline behavior. Prints PASS/WARN/FAIL.
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------
class Status(Enum):
    PASS = "PASS"
    WARN = "WARN"
    FAIL = "FAIL"
    SKIP = "SKIP"


@dataclass
class CheckResult:
    name: str
    status: Status
    detail: str = ""


@dataclass
class Section:
    title: str
    checks: List[CheckResult] = field(default_factory=list)

    def add(self, name: str, status: Status, detail: str = "") -> None:
        self.checks.append(CheckResult(name, status, detail))


# ---------------------------------------------------------------------------
# Section 1 — Data integrity
# ---------------------------------------------------------------------------
def section_data_integrity() -> Section:
    s = Section("1. Data Integrity")
    from src.config import CATEGORIES, PRIORITIES, TICKETS_CSV

    if not TICKETS_CSV.exists():
        s.add("tickets.csv exists", Status.FAIL,
              f"File missing — run '.\\make.ps1 data'")
        return s

    s.add("tickets.csv exists", Status.PASS, str(TICKETS_CSV))

    import pandas as pd
    df = pd.read_csv(TICKETS_CSV, parse_dates=[
        "created_at", "first_response_at", "resolved_at"
    ])

    # Row count
    n = len(df)
    if n >= 1000:
        s.add(f"Row count >= 1000", Status.PASS, f"got {n:,}")
    else:
        s.add(f"Row count too small", Status.WARN, f"got {n}")

    # Schema
    expected_cols = {"ticket_id", "created_at", "first_response_at",
                     "resolved_at", "status", "category", "subcategory",
                     "priority", "description", "requester_id",
                     "assigned_group", "kb_article_id", "sla_target_hours",
                     "resolution_hours", "sla_met"}
    missing = expected_cols - set(df.columns)
    if not missing:
        s.add("All 15 required columns present", Status.PASS)
    else:
        s.add("Schema columns", Status.FAIL, f"missing: {missing}")
        return s

    # Ticket IDs unique
    if df["ticket_id"].is_unique:
        s.add("Ticket IDs unique", Status.PASS)
    else:
        n_dup = len(df) - df["ticket_id"].nunique()
        s.add("Ticket IDs", Status.FAIL, f"{n_dup} duplicates")

    # Categories valid
    invalid = set(df["category"]) - set(CATEGORIES)
    if not invalid:
        s.add("All categories in valid set", Status.PASS)
    else:
        s.add("Category values", Status.FAIL, f"unknown: {invalid}")

    # Category distribution
    mix = df["category"].value_counts(normalize=True)
    expected = {"Software": 0.40, "Access": 0.28, "Hardware": 0.18, "Network": 0.14}
    cat_summary = " | ".join(f"{c}={mix.get(c, 0):.1%}" for c in CATEGORIES)
    deviations = {c: abs(mix.get(c, 0) - p) for c, p in expected.items()}
    max_dev = max(deviations.values())
    if max_dev <= 0.04:
        s.add("Category distribution healthy", Status.PASS, cat_summary)
    elif max_dev <= 0.08:
        worst = max(deviations, key=deviations.get)
        s.add("Category distribution skewed", Status.WARN,
              f"{cat_summary} | worst: {worst} off by {deviations[worst]:.1%}")
    else:
        worst = max(deviations, key=deviations.get)
        s.add("Category distribution wrong", Status.FAIL,
              f"{worst} off by {deviations[worst]:.1%}")

    # Priorities valid
    invalid_pri = set(df["priority"]) - set(PRIORITIES)
    if not invalid_pri:
        s.add("All priorities in valid set", Status.PASS)
    else:
        s.add("Priority values", Status.FAIL, f"unknown: {invalid_pri}")

    # Priority distribution
    pri_mix = df["priority"].value_counts(normalize=True)
    p1_rate = pri_mix.get("P1", 0)
    pri_summary = " | ".join(f"{p}={pri_mix.get(p, 0):.1%}" for p in PRIORITIES)
    if 0.02 <= p1_rate <= 0.08:
        s.add(f"P1 rate realistic (2-8%)", Status.PASS, pri_summary)
    elif 0.01 <= p1_rate <= 0.12:
        s.add(f"P1 rate borderline", Status.WARN, pri_summary)
    else:
        s.add(f"P1 rate unrealistic", Status.FAIL, f"got {p1_rate:.1%}")

    # SLA targets
    sla_map = {"P1": 4.0, "P2": 8.0, "P3": 24.0, "P4": 72.0}
    sla_ok = all((df[df["priority"] == p]["sla_target_hours"] == t).all()
                 for p, t in sla_map.items())
    if sla_ok:
        s.add("SLA targets correctly mapped to priorities", Status.PASS)
    else:
        s.add("SLA target mapping", Status.FAIL, "some priorities have wrong SLA")

    # Status integrity
    resolved = df[df["status"] == "Resolved"]
    if resolved["resolution_hours"].notna().all():
        s.add(f"All {len(resolved):,} resolved tickets have resolution_hours",
              Status.PASS)
    else:
        n_miss = resolved["resolution_hours"].isna().sum()
        s.add("Resolution hours integrity", Status.FAIL,
              f"{n_miss} resolved tickets missing resolution_hours")

    # Time order
    if (resolved["resolved_at"] > resolved["created_at"]).all():
        s.add("All resolved_at > created_at", Status.PASS)
    else:
        s.add("Time order", Status.FAIL,
              "some tickets resolved before they were created")

    return s


# ---------------------------------------------------------------------------
# Section 2 — Ticket-category alignment (descriptions match their labels)
# ---------------------------------------------------------------------------
EXPECTED_KEYWORDS = {
    "Network": [
        # Component-language (original)
        "vpn", "wifi", "network", "dns", "connect", "internet",
        "router", "outage", "globalprotect",
        # Symptom-language (DNS / slow-web templates)
        "websites", "loading", "browsing", "browse", "slow",
        # Outage / reachability templates
        "internal services", "reach", "share drive",
        # Additional VPN vendor terms
        "tunnel", "anyconnect",
    ],
    "Software": ["outlook", "excel", "teams", "app", "software", "crash",
                 "license", "tableau", "servicenow", "install", "sap"],
    "Hardware": ["laptop", "screen", "battery", "printer", "monitor",
                 "mouse", "keyboard", "headset", "dock"],
    "Access": [
        # Password Reset / general access
        "password", "account", "ad ", "mfa", "sso", "okta",
        "lock", "permission", "credential",
        # Group / Role templates (the missing ~39%)
        "access to", "distribution list", "role change", "snowflake",
        "shared folder", "dl-",
    ],
}


def section_ticket_category_alignment() -> Section:
    """
    Verify ticket descriptions actually match their assigned category by
    checking keyword overlap. Catches the case where labels are correct
    in metadata but descriptions are mislabeled (e.g., a swapped templates bug).
    """
    s = Section("2. Ticket-Category Alignment")
    from src.config import TICKETS_CSV

    if not TICKETS_CSV.exists():
        s.add("Skipped — no tickets.csv", Status.SKIP)
        return s

    import pandas as pd
    df = pd.read_csv(TICKETS_CSV, usecols=["category", "description"])
    df["description_lower"] = df["description"].fillna("").str.lower()

    for category, keywords in EXPECTED_KEYWORDS.items():
        cat_df = df[df["category"] == category]
        if cat_df.empty:
            s.add(f"{category}: no tickets", Status.WARN)
            continue

        # Check fraction containing at least one expected keyword
        pattern = "|".join(keywords)
        matches = cat_df["description_lower"].str.contains(pattern, regex=True)
        match_rate = matches.mean()

        if match_rate >= 0.85:
            s.add(f"{category}: descriptions match label",
                  Status.PASS, f"{match_rate:.1%} contain expected keywords")
        elif match_rate >= 0.70:
            s.add(f"{category}: descriptions mostly match label",
                  Status.WARN,
                  f"only {match_rate:.1%} contain expected keywords — noise level may be high")
        else:
            s.add(f"{category}: descriptions don't match label",
                  Status.FAIL,
                  f"only {match_rate:.1%} contain expected keywords — "
                  f"templates may be mislabeled")

    return s


# ---------------------------------------------------------------------------
# Section 3 — Knowledge base
# ---------------------------------------------------------------------------
def section_knowledge_base() -> Section:
    s = Section("3. Knowledge Base")
    from src.config import KB_JSON

    if not KB_JSON.exists():
        s.add("knowledge_base.json exists", Status.FAIL,
              f"File missing — run '.\\make.ps1 kb'")
        return s

    import json
    kb = json.loads(KB_JSON.read_text())

    s.add("knowledge_base.json exists", Status.PASS,
          f"{len(kb)} articles")

    if len(kb) >= 10:
        s.add(f"KB article count adequate", Status.PASS, f"{len(kb)} articles")
    else:
        s.add("KB article count low", Status.WARN, f"{len(kb)} articles")

    required = {"kb_id", "title", "category", "tags", "content"}
    bad = [a for a in kb if not required.issubset(a.keys())]
    if not bad:
        s.add("All articles have required fields", Status.PASS)
    else:
        s.add("KB schema", Status.FAIL,
              f"{len(bad)} articles missing fields")

    # KB IDs unique
    ids = [a["kb_id"] for a in kb]
    if len(ids) == len(set(ids)):
        s.add("All KB IDs unique", Status.PASS)
    else:
        s.add("KB IDs", Status.FAIL, f"{len(ids) - len(set(ids))} duplicates")

    # Content lengths
    short = [a for a in kb if len(a["content"]) < 100]
    if not short:
        s.add("All KB articles have substantive content (≥100 chars)",
              Status.PASS)
    else:
        s.add("KB content length", Status.WARN,
              f"{len(short)} articles under 100 chars")

    # Template→KB linkage
    from src.generate_data import TEMPLATES
    kb_ids = {a["kb_id"] for a in kb}
    template_refs = {t.resolution_kb_id for t in TEMPLATES}
    orphans = template_refs - kb_ids
    if not orphans:
        s.add("All templates link to existing KB articles", Status.PASS)
    else:
        s.add("Template→KB linkage", Status.FAIL,
              f"orphan refs: {orphans}")

    return s


# ---------------------------------------------------------------------------
# Section 4 — Model artifacts
# ---------------------------------------------------------------------------
def section_model_artifacts() -> Section:
    s = Section("4. Model Artifacts")
    from src.classifier import TicketClassifier
    from src.retriever import KBRetriever

    # Classifier
    if TicketClassifier.DEFAULT_MODEL_PATH.exists():
        s.add("classifier.joblib exists", Status.PASS,
              str(TicketClassifier.DEFAULT_MODEL_PATH))
        try:
            clf = TicketClassifier.load()
            if clf.is_trained:
                s.add("Classifier loads and is_trained", Status.PASS)
            else:
                s.add("Classifier", Status.FAIL, "loaded but not marked trained")
        except Exception as exc:
            s.add("Classifier load", Status.FAIL, str(exc))
    else:
        s.add("classifier.joblib missing", Status.FAIL,
              "run '.\\make.ps1 train-quick'")

    # Retriever
    idx_dir = KBRetriever.DEFAULT_INDEX_DIR
    if (idx_dir / KBRetriever.INDEX_FILE).exists():
        s.add("retriever index exists", Status.PASS, str(idx_dir))
        try:
            retriever = KBRetriever.load()
            if retriever.is_built:
                s.add(f"Retriever loads ({retriever.n_articles} articles)",
                      Status.PASS)
            else:
                s.add("Retriever", Status.FAIL, "loaded but not built")
        except Exception as exc:
            s.add("Retriever load", Status.FAIL, str(exc))
    else:
        s.add("retriever index missing", Status.FAIL,
              "run '.\\make.ps1 build-index'")

    return s


# ---------------------------------------------------------------------------
# Section 5 — Classification metrics on a fresh holdout
# ---------------------------------------------------------------------------
def section_classification_metrics() -> Section:
    s = Section("5. Classification Metrics (clean holdout)")

    from src.classifier import TicketClassifier
    if not TicketClassifier.DEFAULT_MODEL_PATH.exists():
        s.add("Skipped — no saved classifier", Status.SKIP)
        return s

    from src.generate_data import generate_tickets
    from src.preprocess import TicketPreprocessor
    from sklearn.metrics import f1_score

    # Build a CLEAN holdout (noise_rate=0) for honest scoring
    holdout = generate_tickets(n_tickets=500, seed=999, noise_rate=0.0)
    prep = TicketPreprocessor()
    holdout = prep.process_dataframe(holdout, text_col="description")

    clf = TicketClassifier.load()
    preds = clf.predict_batch(holdout["description"].tolist())
    pred_cats = [p.category for p in preds]
    pred_pris = [p.priority for p in preds]

    cat_f1 = f1_score(holdout["category"], pred_cats, average="macro")
    pri_f1 = f1_score(holdout["priority"], pred_pris, average="macro")
    cat_acc = sum(p == a for p, a in zip(pred_cats, holdout["category"])) / len(preds)
    pri_acc = sum(p == a for p, a in zip(pred_pris, holdout["priority"])) / len(preds)

    if cat_f1 >= 0.80:
        s.add(f"Category macro F1 >= 0.80", Status.PASS, f"got {cat_f1:.3f}")
    elif cat_f1 >= 0.65:
        s.add(f"Category macro F1 weak", Status.WARN, f"got {cat_f1:.3f}")
    else:
        s.add(f"Category macro F1 too low", Status.FAIL, f"got {cat_f1:.3f}")

    if pri_f1 >= 0.40:
        s.add(f"Priority macro F1 >= 0.40", Status.PASS, f"got {pri_f1:.3f}")
    elif pri_f1 >= 0.30:
        s.add(f"Priority macro F1 borderline", Status.WARN, f"got {pri_f1:.3f}")
    else:
        s.add(f"Priority macro F1 too low", Status.FAIL, f"got {pri_f1:.3f}")

    s.add(f"Category accuracy", Status.PASS, f"{cat_acc:.1%}")
    s.add(f"Priority accuracy", Status.PASS, f"{pri_acc:.1%}")

    # Spot-check obvious examples
    spot_cases = [
        ("VPN keeps disconnecting from corporate network", "Network"),
        ("Outlook application crashes on startup", "Software"),
        ("Laptop battery wont hold charge", "Hardware"),
        ("Locked out of my AD account", "Access"),
    ]
    correct = sum(
        1 for text, expected in spot_cases
        if clf.predict(text).category == expected
    )
    if correct >= 3:
        s.add(f"Obvious examples ≥3/4 correct ({correct}/4)", Status.PASS)
    elif correct >= 2:
        s.add(f"Obvious examples weak ({correct}/4)", Status.WARN)
    else:
        s.add(f"Obvious examples failing ({correct}/4)", Status.FAIL)

    return s


# ---------------------------------------------------------------------------
# Section 6 — Retrieval relevance
# ---------------------------------------------------------------------------
RETRIEVAL_BENCHMARK = [
    ("VPN tunnel keeps dropping connection", "Network"),
    ("Office WiFi keeps disconnecting on third floor", "Network"),
    ("Outlook crashes whenever I open large attachments", "Software"),
    ("Need to install Tableau on my workstation", "Software"),
    ("Laptop battery won't hold charge anymore", "Hardware"),
    ("Printer paper jam keeps recurring", "Hardware"),
    ("Need password reset, locked out of account", "Access"),
    ("SSO is failing for our entire department", "Access"),
]


def section_retrieval_quality() -> Section:
    s = Section("6. Retrieval Relevance (8-query benchmark)")

    from src.retriever import KBRetriever
    idx_dir = KBRetriever.DEFAULT_INDEX_DIR
    if not (idx_dir / KBRetriever.INDEX_FILE).exists():
        s.add("Skipped — no retriever index", Status.SKIP)
        return s

    retriever = KBRetriever.load()

    correct = 0
    total = len(RETRIEVAL_BENCHMARK)
    for query, expected_cat in RETRIEVAL_BENCHMARK:
        results = retriever.search(query, top_k=1)
        if results and results[0].category == expected_cat:
            correct += 1

    pct = correct / total
    if pct >= 0.85:
        s.add(f"Top-1 category match rate ({correct}/{total})",
              Status.PASS, f"{pct:.0%}")
    elif pct >= 0.70:
        s.add(f"Top-1 category match rate ({correct}/{total})",
              Status.WARN, f"{pct:.0%}")
    else:
        s.add(f"Top-1 category match rate too low ({correct}/{total})",
              Status.FAIL, f"{pct:.0%}")

    # Score gap on a clear query
    results = retriever.search("VPN tunnel keeps dropping connection", top_k=2)
    if len(results) >= 2:
        gap = results[0].score - results[1].score
        if gap >= 0.05:
            s.add("Retrieval confidence gap healthy", Status.PASS,
                  f"top={results[0].score:.3f} vs #2={results[1].score:.3f}")
        else:
            s.add("Retrieval confidence gap weak", Status.WARN,
                  f"top-1 only {gap:.3f} above #2")

    return s


# ---------------------------------------------------------------------------
# Section 7 — End-to-end pipeline
# ---------------------------------------------------------------------------
def section_pipeline_e2e() -> Section:
    s = Section("7. End-to-End Pipeline")

    from src.classifier import TicketClassifier
    from src.retriever import KBRetriever
    if not (TicketClassifier.DEFAULT_MODEL_PATH.exists() and
            (KBRetriever.DEFAULT_INDEX_DIR / KBRetriever.INDEX_FILE).exists()):
        s.add("Skipped — missing model artifacts", Status.SKIP)
        return s

    from src.pipeline import TicketTriagePipeline

    pipeline = TicketTriagePipeline()
    start = time.time()
    pipeline.warm_up()
    warm_time = time.time() - start

    if warm_time < 30:
        s.add(f"Pipeline warm-up under 30s", Status.PASS, f"{warm_time:.1f}s")
    else:
        s.add(f"Pipeline warm-up slow", Status.WARN, f"{warm_time:.1f}s")

    # End-to-end on a clear query
    result = pipeline.triage(
        "Major SSO outage affecting our entire finance team this morning"
    )
    if result.priority == "P1":
        s.add("SSO outage triaged as P1", Status.PASS,
              f"conf {result.priority_confidence:.2f}")
    else:
        s.add(f"SSO outage triaged as {result.priority} (expected P1)",
              Status.WARN)

    # Top KB suggestion should be SSO-related
    if result.kb_suggestions and result.kb_suggestions[0]["kb_id"] == "KB-ACC-003":
        s.add("Top KB for SSO outage = KB-ACC-003", Status.PASS,
              f"score {result.kb_suggestions[0]['score']:.3f}")
    else:
        actual = (result.kb_suggestions[0]["kb_id"]
                  if result.kb_suggestions else "(none)")
        s.add(f"Top KB for SSO outage", Status.WARN,
              f"got {actual}, expected KB-ACC-003")

    # Latency on warm pipeline
    start = time.time()
    pipeline.triage("VPN keeps disconnecting")
    latency_ms = (time.time() - start) * 1000
    if latency_ms < 500:
        s.add(f"Warm latency under 500ms", Status.PASS, f"{latency_ms:.0f}ms")
    else:
        s.add(f"Warm latency slow", Status.WARN, f"{latency_ms:.0f}ms")

    return s


# ---------------------------------------------------------------------------
# Section 8 — API health (optional)
# ---------------------------------------------------------------------------
def section_api_health() -> Section:
    s = Section("8. API Health (optional — only if 'serve' is running)")

    try:
        import requests
    except ImportError:
        s.add("Skipped — requests not installed", Status.SKIP)
        return s

    try:
        r = requests.get("http://localhost:8000/health", timeout=2)
        if r.status_code == 200:
            s.add("API /health responding", Status.PASS)
        else:
            s.add("API /health", Status.FAIL, f"status {r.status_code}")
            return s

        r = requests.get("http://localhost:8000/ready", timeout=2)
        body = r.json()
        if body.get("ready"):
            s.add("API /ready reports ready", Status.PASS)
        else:
            s.add("API /ready", Status.WARN, "pipeline not loaded")

        r = requests.post(
            "http://localhost:8000/triage",
            json={"text": "VPN keeps disconnecting"},
            timeout=10,
        )
        if r.status_code == 200:
            body = r.json()
            if body["category"] == "Network":
                s.add("API /triage routes Network correctly", Status.PASS)
            else:
                s.add("API /triage", Status.WARN,
                      f"VPN→{body['category']} (expected Network)")
        else:
            s.add("API /triage", Status.FAIL, f"status {r.status_code}")

    except requests.RequestException:
        s.add("Skipped — API not running on localhost:8000", Status.SKIP)

    return s


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
ICONS = {
    Status.PASS: "[PASS]",
    Status.WARN: "[WARN]",
    Status.FAIL: "[FAIL]",
    Status.SKIP: "[SKIP]",
}


def print_section(section: Section) -> None:
    print()
    print(f"  {section.title}")
    print("  " + "-" * 68)
    for c in section.checks:
        icon = ICONS[c.status]
        line = f"  {icon} {c.name}"
        if c.detail:
            line += f"  ({c.detail})"
        print(line)


def print_summary(sections: List[Section], total_time: float) -> int:
    counts = {s: 0 for s in Status}
    for sec in sections:
        for c in sec.checks:
            counts[c.status] += 1

    total = sum(counts.values())
    print()
    print("=" * 70)
    print("  VALIDATION SUMMARY")
    print("=" * 70)
    print(f"  Total checks:  {total}")
    print(f"  PASS:          {counts[Status.PASS]}")
    print(f"  WARN:          {counts[Status.WARN]}")
    print(f"  FAIL:          {counts[Status.FAIL]}")
    print(f"  SKIP:          {counts[Status.SKIP]}")
    print(f"  Time:          {total_time:.1f}s")
    print("=" * 70)

    if counts[Status.FAIL] > 0:
        print(f"\n  ❌ {counts[Status.FAIL]} CHECK(S) FAILED — fix before pushing.\n")
    elif counts[Status.WARN] > 0:
        print(f"\n  ⚠️  {counts[Status.WARN]} WARNING(S) — review but not blocking.\n")
    else:
        print(f"\n  ✅ ALL CHECKS PASSED — project is in good shape.\n")

    return counts[Status.FAIL]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(description="IntelliDesk validation")
    parser.add_argument("--strict", action="store_true",
                        help="Fail on any WARN, not just FAIL")
    args = parser.parse_args()

    print()
    print("=" * 70)
    print("  INTELLIDESK VALIDATION REPORT")
    print("=" * 70)

    start = time.time()

    sections = []
    runners = [
        section_data_integrity,
        section_ticket_category_alignment,
        section_knowledge_base,
        section_model_artifacts,
        section_classification_metrics,
        section_retrieval_quality,
        section_pipeline_e2e,
        section_api_health,
    ]

    for runner in runners:
        try:
            sec = runner()
        except Exception as exc:
            sec = Section(f"{runner.__name__} (errored)")
            sec.add("Exception during validation", Status.FAIL, str(exc))
        sections.append(sec)
        print_section(sec)

    fail_count = print_summary(sections, time.time() - start)

    # Determine exit code
    if fail_count > 0:
        return 1
    if args.strict:
        warn_count = sum(1 for s in sections for c in s.checks
                         if c.status == Status.WARN)
        if warn_count > 0:
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())