import math
from PIL import Image
from sentence_transformers import SentenceTransformer

from .semantic_search import cosine_similarity
from .search_utils import load_movies

class MultimodalSearch:
    def __init__(self, model_name="clip-ViT-B-32", movies: list[dict] = None):
        self.sentence_tranformer = SentenceTransformer(model_name)
        self.texts = [f"{doc['title']}: {doc['description']}" for doc in movies]
        self.text_embeddings = self.sentence_tranformer.encode(self.texts, show_progress_bar=True)
        self.movies = movies

    def embed_image(self, image_path: str):
        image = Image.open(image_path)
        embedding = self.sentence_tranformer.encode(image)
        return embedding

    def search_with_image(self, image_path: str):
        img_embedding = self.embed_image(image_path)
        img_similarities = [(movie, cosine_similarity(img_embedding, text_embedding)) for text_embedding,movie in
                            zip(self.text_embeddings, self.movies)]
        img_similarities.sort(key=lambda x: x[1], reverse=True)
        return img_similarities[:5]


def verify_image_embedding(image_path: str):
    search = MultimodalSearch()
    embedding = search.embed_image(image_path)
    print(f"Embedding shape: {embedding.shape[0]} dimensions")

def image_search_command(image_path:  str):
    movies = load_movies()
    search = MultimodalSearch(movies=movies)
    results = search.search_with_image(image_path)
    print(f"Image Search Results for {image_path}:")
    for i, (movie, score) in enumerate(results, 1):
        print(f"\n{i}. {movie.get('title')} (similarity: {math.floor(score * 1000) / 1000:.3f})")
        print(f"   {movie.get("description")[:100]}")
