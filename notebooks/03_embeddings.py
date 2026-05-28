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
# # IntelliDesk: Semantic Retrieval & Embeddings
#
# **Goal:** show why semantic retrieval matters for KB suggestions, how the
# embedding space is structured, and why retrieval + classifier together is
# stronger than either alone.

# %%
from pathlib import Path
import sys
import warnings

_PROJECT_ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(_PROJECT_ROOT))
warnings.filterwarnings("ignore")

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams["figure.figsize"] = (10, 6)
plt.rcParams["axes.titleweight"] = "bold"

from src.config import KB_JSON, CATEGORIES, TICKETS_CSV
from src.retriever import KBRetriever
from src.classifier import TicketClassifier

# %% [markdown]
# ## 1. Why semantic retrieval (not keyword search)?
#
# **The keyword search failure mode:**
# A ticket says *"my Outlook keeps freezing when I open the calendar."*
# A keyword-search KB lookup with terms `freeze + outlook + calendar` might
# miss the relevant article *"Outlook Application Hangs and Crashes"* if the
# article uses "hangs" and "stops responding" rather than "freezes."
#
# **Semantic retrieval** maps both query and KB articles into a vector space
# where *"freezing"* and *"hangs"* land near each other regardless of exact
# wording. Same idea behind RAG systems for LLMs.
#
# **The model:** `all-MiniLM-L6-v2` from sentence-transformers. 22MB, ~50ms
# CPU latency, 384-dimensional embeddings. Trained on >1B sentence pairs.

# %% [markdown]
# ## 2. Load the embedding model and KB
#
# We instantiate the same sentence-transformer the retriever uses internally
# (`all-MiniLM-L6-v2`). Encoding is deterministic, so these vectors are
# byte-identical to whatever the retriever has cached.

# %%
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"  # matches src/retriever.py
model = SentenceTransformer(EMBED_MODEL_NAME)

kb = json.loads(KB_JSON.read_text())
print(f"KB articles: {len(kb)}")
print(f"\nCategories represented:")
print(pd.Series([a["category"] for a in kb]).value_counts())

retriever = KBRetriever.load()
print(f"\nRetriever loaded: is_built={retriever.is_built}")

# Encode the same way the retriever does internally
kb_texts = [f"{a['title']}. {a['content']}" for a in kb]
embeddings = model.encode(kb_texts, normalize_embeddings=True)
embeddings = np.array(embeddings)

print(f"\nEmbedding dimension: {embeddings.shape[1]}")
print(f"Articles indexed:    {len(kb)}")
print(f"Embeddings shape:    {embeddings.shape}")

# %% [markdown]
# ## 3. Visualize the embedding space with UMAP
#
# 384 dimensions is too many to inspect directly. UMAP projects them to 2D
# while preserving local neighborhoods, so articles about VPN should cluster
# together, separated from articles about password resets.

# %%
import umap

reducer = umap.UMAP(n_neighbors=5, min_dist=0.3, random_state=42)
embeddings_2d = reducer.fit_transform(embeddings)

viz_df = pd.DataFrame({
    "x": embeddings_2d[:, 0],
    "y": embeddings_2d[:, 1],
    "kb_id": [a["kb_id"] for a in kb],
    "title": [a["title"] for a in kb],
    "category": [a["category"] for a in kb],
})

fig, ax = plt.subplots(figsize=(12, 8))
palette = dict(zip(CATEGORIES, sns.color_palette("muted", n_colors=len(CATEGORIES))))
for cat in CATEGORIES:
    sub = viz_df[viz_df["category"] == cat]
    ax.scatter(sub["x"], sub["y"], label=cat, s=200, alpha=0.7, color=palette[cat])

for _, row in viz_df.iterrows():
    ax.annotate(row["kb_id"], (row["x"], row["y"]),
                fontsize=8, ha="left", va="bottom", xytext=(5, 5),
                textcoords="offset points")

ax.set_title("KB articles in 2D semantic space (UMAP projection)")
ax.set_xlabel("UMAP-1")
ax.set_ylabel("UMAP-2")
ax.legend(title="Category", loc="best")
plt.tight_layout()
plt.show()

# %% [markdown]
# **What we see:** articles cluster by category, with Network and Access in
# distinct regions. The embedding model has learned semantic groupings
# without ever being explicitly told the categories.

# %% [markdown]
# ## 4. Live retrieval examples

# %%
queries = [
    "VPN keeps disconnecting from corporate network",
    "Outlook application crashes when opening attachments",
    "Locked out of my AD account, need urgent reset",
    "Major SSO outage affecting our entire department",
    "Need new license for Tableau Desktop on my workstation",
]

for q in queries:
    print(f"Query: {q!r}")
    results = retriever.search(q, top_k=3)
    for i, r in enumerate(results, 1):
        print(f"  {i}. [{r.kb_id}] {r.title:55s} score={r.score:.3f}")
    print()

# %% [markdown]
# ## 5. Query → KB similarity matrix
#
# How does each query score against every KB article?

# %%
query_embeddings = model.encode(queries, normalize_embeddings=True)
similarities = cosine_similarity(query_embeddings, embeddings)

sim_df = pd.DataFrame(
    similarities,
    index=[f"Q{i+1}" for i in range(len(queries))],
    columns=[a["kb_id"] for a in kb],
)

fig, ax = plt.subplots(figsize=(13, 5))
sns.heatmap(sim_df, annot=True, fmt=".2f", cmap="YlOrRd",
            cbar_kws={"label": "Cosine similarity"}, ax=ax,
            annot_kws={"size": 8})
ax.set_title("Query → KB similarity matrix")
plt.tight_layout()
plt.show()

print("\nQueries:")
for i, q in enumerate(queries, 1):
    print(f"  Q{i}. {q}")

# %% [markdown]
# **What we see:** each query has one or two clear winners (dark cells)
# while most articles score low (light cells). High signal, low noise.

# %% [markdown]
# ## 6. Defense in depth: classifier + retriever together
#
# A real example: the query *"SSO outage affecting finance team"* sometimes
# gets misclassified by the category model as **Software** (because "SSO"
# is rare in training vocabulary). But the **retriever** correctly returns
# **KB-ACC-003 (SSO Outages)** as the top KB.

# %%
ambiguous_query = "SSO outage affecting our finance team this morning"

print(f"Query: {ambiguous_query!r}\n")

clf = TicketClassifier.load()
pred = clf.predict(ambiguous_query)
print(f"Classifier prediction: {pred.category} (conf={pred.category_confidence:.2f}), "
      f"{pred.priority} (conf={pred.priority_confidence:.2f})")

print(f"\nRetriever top-3:")
for i, r in enumerate(retriever.search(ambiguous_query, top_k=3), 1):
    print(f"  {i}. [{r.kb_id}] {r.title} (score={r.score:.3f})")

# %% [markdown]
# Two models, different blind spots. The classifier is fast and good at
# common patterns but weak on rare vocabulary. The retriever uses semantic
# similarity from a 1B-pair-trained model and handles rare vocabulary well.
# The pipeline surfaces both so a wrong classifier prediction doesn't lose
# the right KB.

# %% [markdown]
# ## 7. KB coverage analysis
#
# How well does the current KB cover the query space? A KB with bad coverage
# means low retrieval scores even on legitimate queries: a signal to write
# more articles.

# %%
sample_tickets = pd.read_csv(TICKETS_CSV).sample(100, random_state=42)
top_scores = []
for desc in sample_tickets["description"]:
    results = retriever.search(desc, top_k=1)
    if results:
        top_scores.append(results[0].score)

fig, ax = plt.subplots(figsize=(10, 4.5))
ax.hist(top_scores, bins=20, edgecolor="white", color=sns.color_palette("muted")[0])
ax.axvline(np.mean(top_scores), color="red", linestyle="--",
           label=f"Mean: {np.mean(top_scores):.2f}")
ax.axvline(0.30, color="orange", linestyle="--",
           label="Pipeline threshold: 0.30")
ax.set_title("Top-1 retrieval score distribution (100 random tickets)")
ax.set_xlabel("Cosine similarity to best-matching KB article")
ax.set_ylabel("Count")
ax.legend()
plt.tight_layout()
plt.show()

print(f"Mean top-1 score:   {np.mean(top_scores):.3f}")
print(f"Median top-1 score: {np.median(top_scores):.3f}")
print(f"Tickets below 0.30 threshold (no useful KB match): "
      f"{sum(s < 0.30 for s in top_scores)} / {len(top_scores)}")

# %% [markdown]
# **What this tells us:** if many tickets fall below the 0.30 threshold,
# the KB has **gaps**: common ticket types without any matching article.
# This is the metric to watch when deciding which KB articles to write next.

# %% [markdown]
# ## Summary
#
# 1. **Semantic retrieval beats keyword search** for paraphrased queries:
#    the embedding space groups synonyms together regardless of wording.
# 2. **The KB clusters meaningfully in 2D**: UMAP projection shows category
#    structure emerging from embeddings alone.
# 3. **Retrieval and classifier have different blind spots**: the pipeline
#    returns both.
# 4. **Top-1 score distribution measures KB coverage**: a metric to drive
#    KB authoring priorities.
