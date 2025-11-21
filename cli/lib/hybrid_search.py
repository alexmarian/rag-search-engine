import json
import os

from .keyword_search import InvertedIndex
from .search_utils import load_movies
from .semantic_search import ChunkedSemanticSearch
from ollama import generate

DEFAULT_ALPHA = 0.5
DEFAULT_SEARCH_LIMIT = 5
DEFAULT_RFF_K = 60
MODEL = "gemma3:4b"


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
      brm = merged.get(doc_id, {
        "doc": self.idx.docmap[doc_id],
        "bm25_score": 0.0,
        "sem_score": 0.0,
        "hybrid_score": 0.0,
      })
      brm["bm25_score"] = bm25_scores[i]
      merged[doc_id] = brm
    for i, sr in enumerate(sem_results):
      doc_id = sr["id"]
      srm = merged.get(doc_id, {
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

  def rrf_search(self, query: str, k: int, limit: int):
    bm25_results = self.idx.bm25_search(query, limit * 500)
    sem_results = self.semantic_search.search_chunks(query, limit * 500)

    merged = {}

    for i, br in enumerate(bm25_results, start=1):
      doc_id = br["id"]
      brm = merged.get(doc_id, {
        "doc": self.idx.docmap[doc_id],
        "bm25_rank": 0.0,
        "sem_rank": 0.0,
        "rrf_score": 0.0
      })
      brm["rrf_score"] += rrf_score(i, k)
      brm["bm25_rank"] = i
      merged[doc_id] = brm
    for i, sr in enumerate(sem_results, start=1):
      doc_id = sr["id"]
      srm = merged.get(doc_id, {
        "doc": self.idx.docmap[doc_id],
        "bm25_rank": 0.0,
        "sem_rank": 0.0,
        "rrf_score": 0.0,
      })
      srm["rrf_score"] += rrf_score(i, k)
      srm["sem_rank"] = i
      merged[doc_id] = srm
    results = list(merged.values())
    results.sort(key=lambda x: x["rrf_score"], reverse=True)
    return results[:limit]


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


def weighted_search(query: str, alpha: float,
    limit: int = DEFAULT_SEARCH_LIMIT):
  documents = load_movies()
  return HybridSearch(documents).weighted_search(query, alpha, limit)


def rrf_search(query: str, k: int,
    limit: int = DEFAULT_SEARCH_LIMIT, enhance: str = None, rerank: str = None):
  if enhance is not None:
    query = enhance_query(query, enhance)
  documents = load_movies()
  match rerank:
    case "individual":
      reranked_results=[]
      search_results = HybridSearch(documents).rrf_search(query, k, limit * 5)
      for sr in search_results:
        reranked_results.append(rerank_result(query, sr))
        reranked_results.sort(key=lambda x: x["rerank_score"], reverse=True)
        return reranked_results[:limit]
    case "batch":
      search_results = HybridSearch(documents).rrf_search(query, k, limit * 2)
      ordered_ids = rerank_results(query, search_results)
      print(ordered_ids)
      sr_by_id = {}
      for sr in search_results:
        doc = sr.get("doc", {})
        doc_id = doc.get("id") or sr.get("id")
        print(f"doc_id: {doc_id}")
        if doc_id is not None:
          sr_by_id[doc_id] = sr
      # return reranked results in the order provided by the reranker
      reranked_list = [sr_by_id[i] for i in ordered_ids if i in sr_by_id]
      return reranked_list[:limit]

    case _:
      return HybridSearch(documents).rrf_search(query, k, limit)


def rrf_score(rank: float, k: int = 60):
  return 1 / (k + rank)

def rerank_results(query: str, search_results: list[dict]):
    doc_list_str = json.dumps(search_results)
    prompt = f"""Rank these movies by relevance to the search query.
              Query: "{query}"
              Movies, json array of search results:
              {doc_list_str}
              
              Return ONLY the doc nodes ids in order of relevance (best match first). Return only the valid JSON list, nothing else.
              """
    response = generate(MODEL, prompt)
    arrtxt=response.response.replace("```json", "")
    arttxt = arrtxt.replace("```", "")
    print(arttxt)
    return json.loads(arttxt)
def rerank_result(query, search_result):
    search_result["rerank_score"] = individual_rerank(query,
                                                        search_result.get(
                                                            "doc"))
    return search_result


def individual_rerank(query, doc) -> float:
  prompt = f"""Rate how well this movie matches the search query.

        Query: "{query}"
        Movie: {doc.get("title", "")} - {doc.get("document", "")}

        Consider:
        - Direct relevance to query
        - User intent (what they're looking for)
        - Content appropriateness
        
        Rate 0-10 (10 = perfect match).
        Give me ONLY the number in your response, no other text or explanation.
        
        Score:"""
  response = generate(MODEL, prompt)
  return float(response.response.rstrip())


def enhance_query(query, enhance):
  match enhance:
    case "spell":
      prompt = f"""Fix any spelling errors in this movie search query.  
                  Only correct obvious typos. Don't change correctly spelled words. Return only corrected query. do not change letters case.
              Query: "{query}"
              If no errors, return the original query.
              Corrected:"""
      response = generate(MODEL, prompt)
      print(f"Enhanced query ({enhance}): '{query}' -> '{response.response}'")
      return response.response
    case "rewrite":
      prompt = f"""Rewrite this movie search query to be more specific and searchable.
                    Original: "{query}"
                    Consider:
                  - Common movie knowledge (famous actors, popular films)
                  - Genre conventions (horror = scary, animation = cartoon)
                  - Keep it concise (under 10 words)
                  - It should be a google style search query that's very specific
                  - Don't use boolean logic
                  
                  Examples:
                  
                  - "that bear movie where leo gets attacked" -> "The Revenant Leonardo DiCaprio bear attack"
                  - "movie about bear in london with marmalade" -> "Paddington London marmalade"
                  - "scary movie with bear from few years ago" -> "bear horror movie 2015-2020"
             
                  Rewritten query:"""
      response = generate(MODEL, prompt)
      print(f"Enhanced query ({enhance}): '{query}' -> '{response.response}'")
      return response.response
    case "expand":
      prompt = f"""Expand this movie search query with related terms.
                  Add synonyms and related concepts that might appear in movie descriptions.
                  Keep expansions relevant and focused.
                  This will be appended to the original query.
                  
                  Examples:
                  
                  - "scary bear movie" -> "scary horror grizzly bear movie terrifying film"
                  - "action movie with bear" -> "action thriller bear chase fight adventure"
                  - "comedy with bear" -> "comedy funny bear humor lighthearted"
                  
                  Query: "{query}"
                  Return only the expanded query.
                  """
      response = generate(MODEL, prompt)
      print(f"Enhanced query ({enhance}): '{query}' -> '{response.response}'")
      return response.response
    case _:
      return query
  return query
