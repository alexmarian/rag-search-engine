import argparse
import base64
import mimetypes
from ollama import generate
model = "gemma3:12b"

def main():
    parser = argparse.ArgumentParser(description="Retrieval Augmented Generation CLI")
    parser.add_argument("--image", help="Movie to search")
    parser.add_argument("--query", help="Question to search")

    args = parser.parse_args()
    mime, _ = mimetypes.guess_type(args.image)
    with open(args.image, "rb") as img_file:
        raw = img_file.read()
        image_data = base64.b64encode(raw).decode('ascii')
    prompt = f"""
    Given the included image and text query, rewrite the text query to improve search results from a movie database. Make sure to:
        - Synthesize visual and textual information
        - Focus on movie-specific details (actors, scenes, style, etc.)
        - Return only the rewritten query, without any additional commentary
    """
    response =  generate(prompt=prompt, model=model, images=[image_data], system=args.query)
    print(f"Rewritten query: {response.response.strip()}")
    print(f"Total tokens:    {response.eval_count}")
if __name__ == "__main__":
    main()