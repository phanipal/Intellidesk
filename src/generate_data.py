"""
Synthetic IT Service Desk ticket generator for IntelliDesk.

Produces a realistic dataset of IT tickets with fields needed for:
  - Classification (category, priority)
  - Semantic retrieval (description, resolution)
  - SLA / MTTR dashboards (timestamps, status)
  - Drift monitoring (timestamp-partitioned)

Fully deterministic when seeded (no numpy global state, no uuid).

Usage:
    python -m src.generate_data --n_tickets 10000 --out data/tickets.csv
"""

from __future__ import annotations

import argparse
import logging
import math
import random
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("intellidesk.datagen")


# `weight` controls how often this template is sampled within its category.
# Outage templates get tiny weights (0.05) to reflect that real outages are
# rare events, not 1-in-N within the category.

@dataclass(frozen=True)
class TicketTemplate:
    category: str
    subcategory: str
    descriptions: Tuple[str, ...]
    priority_weights: Tuple[float, float, float, float]  # P1, P2, P3, P4
    resolution_kb_id: str
    weight: float = 1.0  # relative sample frequency within category


TEMPLATES: List[TicketTemplate] = [
    TicketTemplate(
        category="Network",
        subcategory="VPN Connectivity",
        descriptions=(
            "Unable to connect to corporate VPN from home office. Error {err}.",
            "VPN disconnects every {n} minutes and I lose access to internal apps.",
            "GlobalProtect is failing to authenticate since this morning.",
            "Cannot reach internal share drives over VPN — connection times out.",
        ),
        priority_weights=(0.05, 0.35, 0.45, 0.15),
        resolution_kb_id="KB-NET-001",
        weight=1.0,
    ),
    TicketTemplate(
        category="Network",
        subcategory="Office WiFi",
        descriptions=(
            "WiFi in {floor} conference room keeps dropping during calls.",
            "Cannot connect laptop to office WiFi, password prompt loops.",
            "Weak WiFi signal in {floor} — speed test shows under {n} Mbps.",
        ),
        priority_weights=(0.02, 0.20, 0.55, 0.23),
        resolution_kb_id="KB-NET-002",
        weight=1.0,
    ),
    TicketTemplate(
        category="Network",
        subcategory="Site Outage",
        descriptions=(
            "Entire {floor} is unable to reach any internal services — suspected outage.",
            "Network is completely down for our team, nobody can work.",
            "Mass outage reported by {n} users in {floor} simultaneously.",
        ),
        priority_weights=(0.70, 0.25, 0.04, 0.01),
        resolution_kb_id="KB-NET-003",
        weight=0.05,  # outages are rare
    ),
    TicketTemplate(
        category="Network",
        subcategory="DNS / Slow Browsing",
        descriptions=(
            "Websites loading very slowly — internal pages fine, external slow.",
            "DNS resolution failures when accessing {url}.",
            "Intermittent slowness browsing the web over the last {n} hours.",
        ),
        priority_weights=(0.02, 0.15, 0.55, 0.28),
        resolution_kb_id="KB-NET-004",
        weight=1.0,
    ),

    TicketTemplate(
        category="Software",
        subcategory="Application Crash",
        descriptions=(
            "Outlook crashes on launch with error {err}.",
            "Excel freezes when opening large files and has to be force-closed.",
            "Teams client crashes during screen share — stack trace attached.",
            "{app} keeps closing unexpectedly after the latest patch.",
        ),
        priority_weights=(0.03, 0.25, 0.55, 0.17),
        resolution_kb_id="KB-SW-001",
        weight=1.0,
    ),
    TicketTemplate(
        category="Software",
        subcategory="Install / Upgrade",
        descriptions=(
            "Need {app} installed on my workstation for project X.",
            "Software Center install of {app} fails with error {err}.",
            "Please upgrade my {app} license to the latest version.",
        ),
        priority_weights=(0.01, 0.05, 0.34, 0.60),
        resolution_kb_id="KB-SW-002",
        weight=1.0,
    ),
    TicketTemplate(
        category="Software",
        subcategory="License / Activation",
        descriptions=(
            "{app} license expired and I cannot open the application.",
            "License reassignment needed — moving from {app} Basic to Pro.",
            "Activation key not accepted for {app}, getting error {err}.",
        ),
        priority_weights=(0.02, 0.18, 0.50, 0.30),
        resolution_kb_id="KB-SW-003",
        weight=0.6,
    ),
    TicketTemplate(
        category="Software",
        subcategory="Production App Down",
        descriptions=(
            "Tableau server is returning 500 errors for all dashboards.",
            "ServiceNow portal is unreachable for the entire finance team.",
            "SAP login hanging for all users in {floor}.",
        ),
        priority_weights=(0.75, 0.20, 0.04, 0.01),
        resolution_kb_id="KB-SW-004",
        weight=0.05,  # outages are rare
    ),

    TicketTemplate(
        category="Hardware",
        subcategory="Laptop",
        descriptions=(
            "Laptop screen flickering and going black intermittently.",
            "Battery not charging, shows {n}% stuck for hours.",
            "Laptop running hot and fan noise is very loud.",
            "Keyboard keys {key1} and {key2} are unresponsive.",
        ),
        priority_weights=(0.03, 0.30, 0.50, 0.17),
        resolution_kb_id="KB-HW-001",
        weight=1.0,
    ),
    TicketTemplate(
        category="Hardware",
        subcategory="Peripherals",
        descriptions=(
            "Mouse not detected on USB port — tried multiple ports.",
            "External monitor showing no signal when connected via HDMI.",
            "Headset microphone not picked up by any application.",
            "Docking station stopped delivering power to connected devices.",
        ),
        priority_weights=(0.01, 0.15, 0.55, 0.29),
        resolution_kb_id="KB-HW-002",
        weight=1.0,
    ),
    TicketTemplate(
        category="Hardware",
        subcategory="Printer",
        descriptions=(
            "Printer on {floor} showing paper jam that cannot be cleared.",
            "Unable to print — job stays in queue indefinitely.",
            "Printer {n} showing toner low warning and prints are faded.",
        ),
        priority_weights=(0.01, 0.08, 0.50, 0.41),
        resolution_kb_id="KB-HW-003",
        weight=0.7,
    ),

    TicketTemplate(
        category="Access",
        subcategory="Password Reset",
        descriptions=(
            "Locked out of my AD account after too many attempts.",
            "Need password reset for {app} — forgot credentials.",
            "MFA token is not generating codes, cannot sign in anywhere.",
        ),
        priority_weights=(0.05, 0.35, 0.45, 0.15),
        resolution_kb_id="KB-ACC-001",
        weight=1.5,  # most common access ticket
    ),
    TicketTemplate(
        category="Access",
        subcategory="Group / Role",
        descriptions=(
            "Need access to {app} shared folder for project kickoff Monday.",
            "Please add me to the DL-{app}-Users distribution list.",
            "Role change — moving to analytics team, need Snowflake access.",
        ),
        priority_weights=(0.01, 0.10, 0.50, 0.39),
        resolution_kb_id="KB-ACC-002",
        weight=1.0,
    ),
    TicketTemplate(
        category="Access",
        subcategory="Critical Access Outage",
        descriptions=(
            "SSO is failing for our entire department, nobody can work.",
            "All team members locked out of production database — suspect SSO.",
            "Okta not issuing tokens for any of our apps this morning.",
        ),
        priority_weights=(0.78, 0.18, 0.03, 0.01),
        resolution_kb_id="KB-ACC-003",
        weight=0.05,  # outages are rare
    ),
]

FILLERS = {
    "err": ["0x80070002", "ERR_CONN_RESET", "E_FAIL", "0x8024A105", "NETERR-42"],
    "n":   [2, 5, 10, 15, 20, 30, 45, 60, 90],
    "floor": ["Floor 2", "Floor 5 East", "Building B Lobby", "HQ 3rd floor", "Satellite Office"],
    "url": ["salesforce.com", "github.com", "internal.corp/wiki", "confluence.corp"],
    "app": ["Outlook", "Excel", "Teams", "Slack", "Zoom", "Tableau", "Snowflake",
            "Adobe Acrobat", "Visual Studio", "SAP GUI", "ServiceNow"],
    "key1": ["Enter", "Shift", "J", "Ctrl", "Tab", "Backspace"],
    "key2": ["Space", "M", "K", "Esc", "F5", "L"],
}

# Appended to ~12% of tickets to inject realistic ambiguity. The ticket's
# TRUE category is unchanged; this just adds a misleading sentence the way
# real users do (e.g. "Outlook crashed AND VPN was also slow"). Forces the
# classifier to weigh primary signal against noise.
NOISE_FRAGMENTS = {
    "Network": (
        "VPN was unstable beforehand.",
        "WiFi connection has been spotty.",
        "Internet seems slow today.",
        "Network might be involved.",
    ),
    "Software": (
        "Outlook also crashed today.",
        "Application seems sluggish overall.",
        "Software was updated recently.",
        "App might have a bug.",
    ),
    "Hardware": (
        "Laptop running hot lately.",
        "Hardware feels unreliable.",
        "Battery has been weak.",
        "Device makes weird noises.",
    ),
    "Access": (
        "Had to reset password recently.",
        "MFA prompts keep appearing.",
        "Account permissions changed last week.",
        "Login takes longer than usual.",
    ),
}


def _fill(text: str, rng: random.Random) -> str:
    """Replace {tokens} in a template with randomly chosen fillers."""
    for key, pool in FILLERS.items():
        token = "{" + key + "}"
        while token in text:
            text = text.replace(token, str(rng.choice(pool)), 1)
    return text


def _resolve_sla_targets() -> Dict[str, float]:
    """SLA targets in hours per priority."""
    return {"P1": 4.0, "P2": 8.0, "P3": 24.0, "P4": 72.0}


def _sample_resolution_hours(priority: str, rng: random.Random) -> float:
    """
    Sample a realistic resolution time in hours.
    ~75% of tickets meet SLA; rest slip with a fat tail.
    """
    sla = _resolve_sla_targets()[priority]
    if rng.random() < 0.75:
        # within SLA: uniform between 5% and 100% of target
        return rng.uniform(0.05 * sla, sla)
    # SLA breach: lognormal tail
    mu = math.log(sla * 1.5)
    sigma = 0.6
    sample = math.exp(rng.gauss(mu, sigma))
    return max(min(sample, sla * 10.0), sla * 1.05)


def generate_tickets(
    n_tickets: int = 10_000,
    start_date: datetime = datetime(2025, 1, 1),
    end_date: datetime = datetime(2026, 4, 1),
    seed: int = 42,
    noise_rate: float = 0.12,
) -> pd.DataFrame:
    """
    Generate a synthetic IT ticket dataset.

    Args:
        n_tickets: Number of tickets to generate.
        start_date: Earliest ticket creation time.
        end_date: Latest ticket creation time.
        seed: RNG seed for full reproducibility.

    Returns:
        DataFrame with one row per ticket, sorted by created_at.
    """
    rng = random.Random(seed)

    # Skewed to resemble real enterprise IT volumes.
    category_weights = {"Software": 0.40, "Access": 0.28, "Hardware": 0.18, "Network": 0.14}

    templates_by_cat: Dict[str, List[TicketTemplate]] = {}
    for t in TEMPLATES:
        templates_by_cat.setdefault(t.category, []).append(t)

    total_seconds = int((end_date - start_date).total_seconds())
    priorities = ("P1", "P2", "P3", "P4")
    sla_map = _resolve_sla_targets()

    rows = []
    for i in range(n_tickets):
        category = rng.choices(
            list(category_weights.keys()),
            weights=list(category_weights.values()),
            k=1,
        )[0]
        templates_in_cat = templates_by_cat[category]
        template_weights = [t.weight for t in templates_in_cat]
        tmpl = rng.choices(templates_in_cat, weights=template_weights, k=1)[0]

        priority = rng.choices(priorities, weights=tmpl.priority_weights, k=1)[0]

        created_at = start_date + timedelta(seconds=rng.randint(0, total_seconds))
        res_hours = _sample_resolution_hours(priority, rng)
        first_response_hours = min(res_hours, rng.uniform(0.05, sla_map[priority] * 0.4))

        # 92% resolved, 6% in progress, 2% open
        status_roll = rng.random()
        if status_roll < 0.92:
            status = "Resolved"
            resolved_at = created_at + timedelta(hours=res_hours)
            first_response_at = created_at + timedelta(hours=first_response_hours)
        elif status_roll < 0.98:
            status = "In Progress"
            resolved_at = None
            first_response_at = created_at + timedelta(hours=first_response_hours)
        else:
            status = "Open"
            resolved_at = None
            first_response_at = None

        description = _fill(rng.choice(tmpl.descriptions), rng)

        if noise_rate > 0 and rng.random() < noise_rate:
            other_categories = [c for c in NOISE_FRAGMENTS if c != tmpl.category]
            noise_cat = rng.choice(other_categories)
            noise_fragment = rng.choice(NOISE_FRAGMENTS[noise_cat])
            description = f"{description} {noise_fragment}"

        rows.append({
            "ticket_id": f"INC{10000000 + i:08d}",
            "created_at": created_at,
            "first_response_at": first_response_at,
            "resolved_at": resolved_at,
            "status": status,
            "category": tmpl.category,
            "subcategory": tmpl.subcategory,
            "priority": priority,
            "description": description,
            "requester_id": f"U{rng.randint(10000, 99999)}",
            "assigned_group": f"{tmpl.category}-Tier{rng.choice([1, 2, 3])}",
            "kb_article_id": tmpl.resolution_kb_id,
            "sla_target_hours": sla_map[priority],
            "resolution_hours": res_hours if status == "Resolved" else None,
            "sla_met": (res_hours <= sla_map[priority]) if status == "Resolved" else None,
        })

    df = pd.DataFrame(rows).sort_values("created_at").reset_index(drop=True)
    logger.info("Generated %d tickets spanning %s → %s",
                len(df), df["created_at"].min(), df["created_at"].max())
    return df


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate synthetic IT ticket data.")
    p.add_argument("--n_tickets", type=int, default=10_000)
    p.add_argument("--out", type=Path, default=Path("data/tickets.csv"))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--noise_rate", type=float, default=0.12,
                   help="Fraction of tickets that get cross-category noise (0.0-1.0)")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df = generate_tickets(
        n_tickets=args.n_tickets,
        seed=args.seed,
        noise_rate=args.noise_rate,
    )
    df.to_csv(args.out, index=False)
    logger.info("Saved dataset → %s", args.out)


if __name__ == "__main__":
    main()
