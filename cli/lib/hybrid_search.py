import os

from .keyword_search import InvertedIndex
from .search_utils import load_movies
from .semantic_search import ChunkedSemanticSearch

DEFAULT_ALPHA = 0.5
DEFAULT_SEARCH_LIMIT = 5


class HybridSearch:
  def __init__(self, documents):
    self.documents = documents
    self.semantic_search = ChunkedSemanticSearch()
    self.semantic_search.load_or_create_chunk_embeddings(documents)
    self.idx = InvertedIndex()
    if not os.path.exists(self.idx.index_path):
      self.idx.build()
      self.idx.save()
    self.idx.load()

  def _bm25_search(self, query, limit):
    self.idx.load()
    return self.idx.bm25_search(query, limit)

  def weighted_search(self, query: str, alpha: float, limit: int):
    bm25_results = self.idx.bm25_search(query, limit * 500)
    sem_results = self.semantic_search.search_chunks(query, limit * 500)

    bm25_scores = normalize_scores([br["score"] for br in bm25_results])
    sem_scores = normalize_scores([sr["score"] for sr in sem_results])

    merged = {}

    for i, br in enumerate(bm25_results):
      doc_id = br["id"]
      brm=merged.get(doc_id, {
        "doc": self.idx.docmap[doc_id],
        "bm25_score": 0.0,
        "sem_score": 0.0,
        "hybrid_score": 0.0,
      })
      brm["bm25_score"] = bm25_scores[i]
      merged[doc_id] = brm
    for i, sr in enumerate(sem_results):
      doc_id = sr["id"]
      srm=merged.get(doc_id, {
        "doc": self.idx.docmap[doc_id],
        "bm25_score": 0.0,
        "sem_score": 0.0,
        "hybrid_score": 0.0,
      })
      srm["sem_score"] = sem_scores[i]
      merged[doc_id] = srm
    results = []
    for doc_id, entry in merged.items():
      entry["hybrid_score"] = hybrid_score(
          entry["bm25_score"], entry["sem_score"], alpha
      )
      results.append(entry)

    results.sort(key=lambda x: x["hybrid_score"], reverse=True)
    return results[:limit]

  def rrf_search(self, query, k, limit=10):
    raise NotImplementedError("RRF hybrid search is not implemented yet.")


def normalize_scores(scores: list[float]) -> list[float]:
  if not scores:
    return []

  min_score = min(scores)
  max_score = max(scores)

  if max_score == min_score:
    return [1.0] * len(scores)

  normalized_scores = []
  for s in scores:
    normalized_scores.append((s - min_score) / (max_score - min_score))

  return normalized_scores


def hybrid_score(bm25_score, semantic_score, alpha=0.5):
  return alpha * bm25_score + (1 - alpha) * semantic_score


def weighted_search(query: str, alpha: float, limit: int = DEFAULT_SEARCH_LIMIT):
  documents = load_movies()
  return HybridSearch(documents).weighted_search(query, alpha, limit)
