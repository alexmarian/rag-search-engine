import argparse
from lib.hybrid_search import (normalize_scores)


def main() -> None:
  parser = argparse.ArgumentParser(description="Hybrid Search CLI")
  subparser = parser.add_subparsers(dest="command", help="Available commands")

  normalize_parser = subparser.add_parser("normalize", help="Normalize a numeric vector")
  normalize_parser.add_argument("vector", nargs="+", type=float, help="Hybrid search vector (space-separated floats)")

  args = parser.parse_args()

  match args.command:
    case "normalize":
      normalized = normalize_scores(args.scores)
      for score in normalized:
        print(f"* {score:.4f}")
    case _:
      parser.print_help()


if __name__ == "__main__":
  main()
