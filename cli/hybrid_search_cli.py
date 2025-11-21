import argparse
from lib.hybrid_search import (normalize_scores, weighted_search, rrf_search,
                               DEFAULT_SEARCH_LIMIT, DEFAULT_ALPHA)


def main() -> None:
  parser = argparse.ArgumentParser(description="Hybrid Search CLI")
  subparser = parser.add_subparsers(dest="command", help="Available commands")

  normalize_parser = subparser.add_parser("normalize",
                                          help="Normalize a numeric vector")
  normalize_parser.add_argument("scores", nargs="+", type=float,
                                help="Hybrid search vector (space-separated floats)")

  weighted_parser = subparser.add_parser("weighted-search",
                                         help="Available commands")
  weighted_parser.add_argument("query", type=str, help="search value")
  weighted_parser.add_argument("--alpha", type=float, nargs="?",
                               default=DEFAULT_ALPHA, help=" alpha value")
  weighted_parser.add_argument("--limit", type=int, nargs="?",
                               default=DEFAULT_SEARCH_LIMIT,
                               help="search limit")

  rrf_parser = subparser.add_parser("rrf-search",
                                    help="Available commands")
  rrf_parser.add_argument("query", type=str, help="search value")
  rrf_parser.add_argument("--k", type=int, nargs="?",
                          default=DEFAULT_ALPHA, help=" alpha value")
  rrf_parser.add_argument("--limit", type=int, nargs="?",
                          default=DEFAULT_SEARCH_LIMIT,
                          help="search limit")
  rrf_parser.add_argument(
      "--enhance",
      type=str,
      choices=["spell","rewrite"],
      help="Query enhancement method",
  )
  args = parser.parse_args()

  match args.command:
    case "rrf-search":
      results = rrf_search(args.query, args.k, args.limit, args.enhance)
      for i, res in enumerate(results, start=1):
        print(f"{i}. {res["doc"]["title"]}")
        print(f"RRF Score: {res["rrf_score"]:.4f}")
        print(
            f"BM25 Rank: {res["bm25_rank"]:.4f}, Semantic Rank: {res["sem_rank"]:.4f}")
        print(res["doc"]["description"][:100])
    case "weighted-search":
      results = weighted_search(args.query, args.alpha, args.limit)
      for i, res in enumerate(results, start=1):
        print(f"{i}. {res["doc"]["title"]}")
        print(f"Hybrid Score: {res["hybrid_score"]:.4f}")
        print(
            f"BM25: {res["bm25_score"]:.4f}, Semantic: {res["sem_score"]:.4f}")
        print(res["doc"]["description"][:100])
    case "normalize":
      normalized = normalize_scores(args.scores)
      for score in normalized:
        print(f"* {score:.4f}")
    case _:
      parser.print_help()


if __name__ == "__main__":
  main()
