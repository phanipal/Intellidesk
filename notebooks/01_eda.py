# ---
# jupyter:
#   jupytext:
#     formats: py:percent,ipynb
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # IntelliDesk: Exploratory Data Analysis
#
# **Goal:** understand the IT ticket dataset before any modeling.
#
# This notebook walks through ticket volume, category and priority distributions,
# SLA performance, resolution times, and the operational patterns a service desk
# manager would care about. Every chart is paired with a "so what": the analytical
# observation a stakeholder takes away.
#
# **Audience:** an IT director or hiring manager who wants to know whether the
# data is well-understood before trusting any model built on top of it.

# %%
from pathlib import Path
import sys

# Make the src package importable when this notebook lives in notebooks/
_PROJECT_ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(_PROJECT_ROOT))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams["figure.figsize"] = (11, 5)
plt.rcParams["axes.titlesize"] = 13
plt.rcParams["axes.titleweight"] = "bold"

from src.config import TICKETS_CSV, CATEGORIES, PRIORITIES

# %% [markdown]
# ## 1. Load the data
#
# 10,000 synthetic tickets generated to mirror real IT service desk patterns:
# realistic SLA distributions, priority mix typical of mid-market IT operations
# (~4% P1, ~50% P3), and intentional noise contamination so models trained on
# this data have to handle ambiguity.

# %%
df = pd.read_csv(
    TICKETS_CSV,
    parse_dates=["created_at", "first_response_at", "resolved_at"],
)
df["description_length"] = df["description"].str.len()
df["created_date"] = df["created_at"].dt.date

print(f"Rows:    {len(df):,}")
print(f"Columns: {df.shape[1]}")
print(f"Date range: {df['created_at'].min().date()} → {df['created_at'].max().date()}")
df.head(3)

# %% [markdown]
# ## 2. Category distribution
#
# Categories follow a Pareto-ish distribution typical of corporate IT: software
# issues dominate, followed by access management. Network and hardware are
# meaningful but smaller buckets.

# %%
cat_counts = df["category"].value_counts().reindex(CATEGORIES)

fig, ax = plt.subplots(figsize=(9, 4.5))
bars = ax.bar(cat_counts.index, cat_counts.values, color=sns.color_palette("muted"))
ax.set_title("Ticket volume by category")
ax.set_ylabel("Tickets")
for bar, value in zip(bars, cat_counts.values):
    pct = value / len(df) * 100
    ax.text(bar.get_x() + bar.get_width() / 2, value + 50,
            f"{value:,}\n({pct:.1f}%)", ha="center", fontsize=10)
ax.set_ylim(0, cat_counts.max() * 1.15)
plt.tight_layout()
plt.show()

# %% [markdown]
# **So what:** Software is 40% of intake. Any improvement to software-issue
# routing has 4x the impact of the same improvement applied to network issues.
# This drives where to invest classifier accuracy.

# %% [markdown]
# ## 3. Priority distribution
#
# Severe class imbalance: P1 is intentionally rare (real outages) while P3
# dominates (routine work). This imbalance is **the** central modeling
# challenge for this dataset.

# %%
pri_counts = df["priority"].value_counts().reindex(PRIORITIES)

fig, ax = plt.subplots(figsize=(9, 4.5))
colors = ["#d62728", "#ff7f0e", "#2ca02c", "#1f77b4"]
bars = ax.bar(pri_counts.index, pri_counts.values, color=colors)
ax.set_title("Ticket volume by priority — note the 12:1 imbalance between P3 and P1")
ax.set_ylabel("Tickets")
for bar, value in zip(bars, pri_counts.values):
    pct = value / len(df) * 100
    ax.text(bar.get_x() + bar.get_width() / 2, value + 50,
            f"{value:,}\n({pct:.1f}%)", ha="center", fontsize=10)
plt.tight_layout()
plt.show()

print(f"\nClass imbalance ratio (P3/P1): {pri_counts['P3'] / pri_counts['P1']:.1f}:1")

# %% [markdown]
# **So what:** A model trained without class-weight calibration would learn
# that "always predict P3" is 50% accurate. The classifier in this project uses
# **square-root scaled balanced weights**. Full balanced weighting actually
# *hurt* macro-F1 by over-correcting.

# %% [markdown]
# ## 4. Daily ticket volume

# %%
daily = df.groupby("created_date").size()

fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(daily.index, daily.values, linewidth=1.0, alpha=0.8)
ax.fill_between(daily.index, daily.values, alpha=0.15)
ax.set_title("Daily ticket volume")
ax.set_ylabel("Tickets per day")
plt.xticks(rotation=30)
plt.tight_layout()
plt.show()

print(f"Mean daily volume:   {daily.mean():.0f} tickets")
print(f"Median daily volume: {daily.median():.0f} tickets")
print(f"Std deviation:       {daily.std():.1f} tickets")

# %% [markdown]
# ## 5. Category × Priority heatmap
#
# Where do severe tickets cluster? This is the chart an IT director uses to
# decide which queue gets dedicated on-call coverage.

# %%
pivot = df.pivot_table(
    index="category", columns="priority", values="ticket_id",
    aggfunc="count", fill_value=0,
).reindex(index=CATEGORIES, columns=PRIORITIES)

fig, ax = plt.subplots(figsize=(8, 4.5))
sns.heatmap(pivot, annot=True, fmt="d", cmap="Blues",
            cbar_kws={"label": "Tickets"}, ax=ax)
ax.set_title("Ticket count: category × priority")
plt.tight_layout()
plt.show()

# %% [markdown]
# **So what:** Access has a disproportionately high P1 count given its volume.
# The "Critical Access Outage" subcategory (SSO outages, mass account lockouts)
# is rare but severe. This is where automated triage saves the most hours.

# %% [markdown]
# ## 6. Resolution time analysis (MTTR)

# %%
resolved = df[df["status"] == "Resolved"].copy()

fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))

sns.boxplot(data=resolved, x="priority", y="resolution_hours",
            order=PRIORITIES, ax=axes[0], showfliers=False)
axes[0].set_title("Resolution hours by priority (outliers hidden)")
axes[0].set_xlabel("")

mttr_by_cat = resolved.groupby("category")["resolution_hours"].median().reindex(CATEGORIES)
axes[1].bar(mttr_by_cat.index, mttr_by_cat.values, color=sns.color_palette("muted"))
axes[1].set_title("Median MTTR by category (hours)")
axes[1].set_ylabel("Hours")
for i, v in enumerate(mttr_by_cat.values):
    axes[1].text(i, v + 0.3, f"{v:.1f}h", ha="center", fontsize=10)

plt.tight_layout()
plt.show()

# %% [markdown]
# ## 7. SLA performance

# %%
sla_by_pri = resolved.groupby("priority").agg(
    sla_met_rate=("sla_met", "mean"),
    n_tickets=("sla_met", "size"),
).reindex(PRIORITIES)

fig, ax = plt.subplots(figsize=(9, 4.5))
bars = ax.bar(sla_by_pri.index, sla_by_pri["sla_met_rate"] * 100,
              color=["#d62728", "#ff7f0e", "#2ca02c", "#1f77b4"])
ax.set_title("SLA met rate by priority")
ax.set_ylabel("% tickets resolved within SLA target")
ax.axhline(95, color="gray", linestyle="--", alpha=0.6, label="95% target")
ax.legend()
ax.set_ylim(0, 105)
for bar, rate, n in zip(bars, sla_by_pri["sla_met_rate"], sla_by_pri["n_tickets"]):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
            f"{rate:.1%}\n(n={n:,})", ha="center", fontsize=9)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 8. Description length per category

# %%
fig, ax = plt.subplots(figsize=(11, 4))
sns.violinplot(data=df, x="category", y="description_length",
               order=CATEGORIES, ax=ax, inner="quartile")
ax.set_title("Description length by category")
ax.set_ylabel("Characters")
ax.set_ylim(0, df["description_length"].quantile(0.99))
plt.tight_layout()
plt.show()

print("\nMean description length by category:")
print(df.groupby("category")["description_length"].mean().round(1).to_string())

# %% [markdown]
# ## 9. Top assigned groups

# %%
top_groups = df["assigned_group"].value_counts().head(10)

fig, ax = plt.subplots(figsize=(10, 4.5))
ax.barh(top_groups.index[::-1], top_groups.values[::-1], color=sns.color_palette("muted"))
ax.set_title("Top 10 assigned groups by ticket volume")
ax.set_xlabel("Tickets")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Key findings
#
# 1. **Software dominates intake at 40%**: primary place to invest in
#    classifier accuracy and KB coverage.
# 2. **Severe priority imbalance (~12:1)**: P3:P1 ratio requires careful
#    sample-weight calibration; full balancing over-corrects.
# 3. **Access has disproportionate P1 share**: SSO outages and mass lockouts
#    are rare but severe. Automated triage saves most hours here.
# 4. **MTTR scales correctly with priority**: sanity check that synthetic data
#    resembles real-world IT operations.
# 5. **SLA met rate >90% across priorities**: healthy baseline; production
#    monitoring should alert when any priority drops below 90%.
#
# Next: notebook 02 tackles classifier design, including the imbalance story
# and how sqrt-scaled weighting outperformed full balancing.
