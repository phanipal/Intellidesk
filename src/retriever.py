"""
Semantic KB retriever using sentence-transformers and FAISS.

Embeds KB articles into a FAISS index and returns top-k matches for a ticket
description.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from src.config import KB_JSON, MODELS_DIR

logger = logging.getLogger("intellidesk.retriever")


@dataclass(frozen=True)
class RetrievalResult:
    """One retrieved KB article with similarity score."""
    kb_id: str
    title: str
    category: str
    content: str
    tags: List[str]
    score: float

    def to_dict(self) -> Dict:
        return asdict(self)


class KBRetriever:
    """
    Semantic retriever over the IntelliDesk knowledge base.

    Lazy-loads the embedding model on first use. After build_index() or
    load(), supports search() and search_batch().
    """

    DEFAULT_MODEL = "all-MiniLM-L6-v2"
    DEFAULT_INDEX_DIR = MODELS_DIR / "retriever"
    INDEX_FILE = "faiss.index"
    METADATA_FILE = "metadata.json"

    def __init__(self, model_name: str = DEFAULT_MODEL):
        self.model_name = model_name
        self._model = None
        self._index = None
        self._metadata: List[Dict] = []  # KB articles parallel to embeddings

    @property
    def is_built(self) -> bool:
        return self._index is not None and len(self._metadata) > 0

    @property
    def n_articles(self) -> int:
        return len(self._metadata)

    def _load_model(self):
        if self._model is not None:
            return self._model
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers is required. "
                "Install with: pip install sentence-transformers"
            ) from exc

        logger.info(
            "Loading embedding model: %s (first use downloads ~22MB, then cached)",
            self.model_name,
        )
        self._model = SentenceTransformer(self.model_name)
        return self._model

    def _encode(self, texts: List[str]) -> np.ndarray:
        """Embed texts and L2-normalize so inner product = cosine similarity."""
        model = self._load_model()
        embeddings = model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return embeddings.astype("float32")

    @staticmethod
    def _article_to_text(article: Dict) -> str:
        """Combine title + content into a single embedding input."""
        return f"{article['title']}. {article['content']}"

    def build_index(self, kb_articles: List[Dict]) -> None:
        """Build FAISS index from list of KB article dicts."""
        try:
            import faiss
        except ImportError as exc:
            raise ImportError(
                "faiss-cpu required. Install with: pip install faiss-cpu"
            ) from exc

        if not kb_articles:
            raise ValueError("Cannot build index from empty KB")

        logger.info("Building FAISS index from %d KB articles", len(kb_articles))
        texts = [self._article_to_text(a) for a in kb_articles]
        embeddings = self._encode(texts)

        dim = embeddings.shape[1]
        self._index = faiss.IndexFlatIP(dim)  # IP on normalized = cosine
        self._index.add(embeddings)
        self._metadata = list(kb_articles)
        logger.info("Index built: %d vectors of dim %d", self._index.ntotal, dim)

    def build_from_json(self, path: Optional[Path] = None) -> None:
        """Build index by loading KB articles from a JSON file."""
        path = path or KB_JSON
        if not path.exists():
            raise FileNotFoundError(f"KB file not found: {path}")
        articles = json.loads(path.read_text())
        self.build_index(articles)

    def search(self, query: str, top_k: int = 3) -> List[RetrievalResult]:
        """Search the KB for articles most similar to the query."""
        if not self.is_built:
            raise RuntimeError("Index not built. Call build_index() or load() first.")

        query = query or ""
        if not query.strip():
            return []

        query_emb = self._encode([query])
        k = min(top_k, self.n_articles)
        scores, indices = self._index.search(query_emb, k)

        results: List[RetrievalResult] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:  # FAISS returns -1 for missing matches
                continue
            article = self._metadata[idx]
            results.append(RetrievalResult(
                kb_id=article["kb_id"],
                title=article["title"],
                category=article["category"],
                content=article["content"],
                tags=article.get("tags", []),
                score=float(score),
            ))
        return results

    def search_batch(
        self, queries: List[str], top_k: int = 3
    ) -> List[List[RetrievalResult]]:
        """Search the KB for multiple queries in one call."""
        if not self.is_built:
            raise RuntimeError("Index not built. Call build_index() or load() first.")

        # Empty queries get empty result lists; only embed non-empty ones
        non_empty_idx = [i for i, q in enumerate(queries) if q and q.strip()]
        if not non_empty_idx:
            return [[] for _ in queries]

        embeddings = self._encode([queries[i] for i in non_empty_idx])
        k = min(top_k, self.n_articles)
        scores_arr, indices_arr = self._index.search(embeddings, k)

        all_results: List[List[RetrievalResult]] = [[] for _ in queries]
        for batch_pos, original_pos in enumerate(non_empty_idx):
            results: List[RetrievalResult] = []
            for score, idx in zip(scores_arr[batch_pos], indices_arr[batch_pos]):
                if idx < 0:
                    continue
                article = self._metadata[idx]
                results.append(RetrievalResult(
                    kb_id=article["kb_id"],
                    title=article["title"],
                    category=article["category"],
                    content=article["content"],
                    tags=article.get("tags", []),
                    score=float(score),
                ))
            all_results[original_pos] = results
        return all_results

    def save(self, dir_path: Optional[Path] = None) -> Path:
        """Save the FAISS index + metadata to disk."""
        try:
            import faiss
        except ImportError as exc:
            raise ImportError("faiss-cpu required") from exc

        if not self.is_built:
            raise RuntimeError("Cannot save an unbuilt retriever")

        dir_path = dir_path or self.DEFAULT_INDEX_DIR
        dir_path.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self._index, str(dir_path / self.INDEX_FILE))
        (dir_path / self.METADATA_FILE).write_text(
            json.dumps({
                "model_name": self.model_name,
                "articles": self._metadata,
            }, indent=2)
        )
        logger.info("Saved retriever -> %s", dir_path)
        return dir_path

    @classmethod
    def load(cls, dir_path: Optional[Path] = None) -> "KBRetriever":
        """Load a saved retriever from disk."""
        try:
            import faiss
        except ImportError as exc:
            raise ImportError("faiss-cpu required") from exc

        dir_path = dir_path or cls.DEFAULT_INDEX_DIR
        idx_path = dir_path / cls.INDEX_FILE
        meta_path = dir_path / cls.METADATA_FILE

        if not idx_path.exists() or not meta_path.exists():
            raise FileNotFoundError(
                f"Retriever artifacts not found in {dir_path}"
            )

        meta = json.loads(meta_path.read_text())
        retriever = cls(model_name=meta["model_name"])
        retriever._index = faiss.read_index(str(idx_path))
        retriever._metadata = meta["articles"]
        logger.info("Loaded retriever <- %s (%d articles)",
                    dir_path, retriever.n_articles)
        return retriever


def main() -> None:
    """CLI: build retriever index from data/knowledge_base.json and save it."""
    import argparse

    parser = argparse.ArgumentParser(description="Build the KB retriever.")
    parser.add_argument("--kb", type=str, default=str(KB_JSON))
    parser.add_argument("--out", type=str,
                        default=str(KBRetriever.DEFAULT_INDEX_DIR))
    parser.add_argument("--query", type=str,
                        help="Optional: run a sample query after building")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    retriever = KBRetriever()
    retriever.build_from_json(Path(args.kb))
    retriever.save(Path(args.out))

    if args.query:
        print("\n" + "=" * 70)
        print(f"  QUERY: {args.query}")
        print("=" * 70)
        for i, r in enumerate(retriever.search(args.query, top_k=3), 1):
            print(f"\n  {i}. [{r.kb_id}] {r.title}")
            print(f"     Category: {r.category}  |  Score: {r.score:.3f}")
            print(f"     {r.content[:140]}...")
        print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
