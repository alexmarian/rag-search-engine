import argparse

from lib.augmented_generation import (augmented_summarization, augmented_generation, augmented_citations, augmented_question_answering)


def main():
    parser = argparse.ArgumentParser(description="Retrieval Augmented Generation CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    rag_parser = subparsers.add_parser(
        "rag", help="Perform RAG (search + generate answer)"
    )
    rag_parser.add_argument("query", type=str, help="Search query for RAG")

    summarize_parser = subparsers.add_parser(
        "summarize", help="Perform RAG (search + summarize answer)"
    )
    summarize_parser.add_argument("query", type=str, help="Search query for RAG")

    summarize_parser.add_argument(
        "--limit", type=int, default=5, help="Number of results to return (default=5)"
    )

    citations_parser = subparsers.add_parser(
        "citations", help="Perform RAG (search + summarize answer with citations)"
    )
    citations_parser.add_argument("query", type=str, help="Search query for RAG")

    citations_parser.add_argument(
    "--limit", type=int, default=5, help="Number of results to return (default=5)")

    question_parser = subparsers.add_parser(
        "question", help="Perform RAG (search + answer the question)"
    )
    question_parser.add_argument("question", type=str, help="Question for RAG")

    question_parser.add_argument(
        "--limit", type=int, default=5, help="Number of results to return (default=5)"
    )

    args = parser.parse_args()

    match args.command:
        case "rag":
            query = args.query
            augmented_generation(query)
        case "summarize":
            query = args.query
            limit = args.limit
            augmented_summarization(query,limit)
        case "citations":
            query = args.query
            limit = args.limit
            augmented_citations(query,limit)
        case "question":
            question = args.question
            limit = args.limit
            augmented_question_answering(question,limit)
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()