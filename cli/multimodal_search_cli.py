import argparse
from lib.multimodal_search import (verify_image_embedding, image_search_command)

def main():
    parser = argparse.ArgumentParser(description="Retrieval Augmented Generation CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    vim_parser = subparsers.add_parser(
        "verify_image_embedding", help="verify image embedding"
    )
    vim_parser.add_argument("image_path", type=str, help="Search image")

    is_parser = subparsers.add_parser(
        "image_search", help="search by an image"
    )
    is_parser.add_argument("image_path", type=str, help="Search image")
    args = parser.parse_args()

    match args.command:
        case "verify_image_embedding":
            image_path = args.image_path
            verify_image_embedding(image_path)
        case "image_search":
            image_path = args.image_path
            image_search_command(image_path)
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()