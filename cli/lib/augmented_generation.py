import json

from .hybrid_search import HybridSearch
from .search_utils import load_movies
from ollama import generate


model = "gpt-oss:20b"
# model = "qwen3-coder:30b"

def augmented_generation(query: str, limit: int = 5):

    hybrid_search = HybridSearch(load_movies())
    results = hybrid_search.rrf_search(query,60,limit)
    docs = json.dumps(results)

    prompt= f"""Answer the question or provide information based on the provided documents. This should be tailored to Hoopla users. Hoopla is a movie streaming service.

Query: {query}

Documents:
{docs}

Provide a comprehensive answer that addresses the query:"""
    response = generate(model, prompt)
    print("Search Results:")
    for result in results:
        title = result.get("title", "")
        print(f"  - {title}")
    print("RAG Response:")
    print(response.response)

def augmented_summarization(query: str, limit: int = 5):

    hybrid_search = HybridSearch(load_movies())
    results = hybrid_search.rrf_search(query,60,limit)
    docs = json.dumps(results)

    prompt= f"""
Provide information useful to this query by synthesizing information from multiple search results in detail.
The goal is to provide comprehensive information so that users know what their options are.
Your response should be information-dense and concise, with several key pieces of information about the genre, plot, etc. of each movie.
This should be tailored to Hoopla users. Hoopla is a movie streaming service.
Query: {query}
Search Results:
{docs}
Provide a comprehensive 3–4 sentence answer that combines information from multiple sources:
"""
    response = generate(model, prompt)
    print("Search Results:")
    for result in results:
        title = result.get("title", "")
        print(f"  - {title}")
    print("LLM Summary:")
    print(response.response)

def augmented_citations(query: str, limit: int = 5):

    hybrid_search = HybridSearch(load_movies())
    results = hybrid_search.rrf_search(query,60,limit)
    docs = json.dumps(results)

    prompt= f"""Answer the question or provide information based on the provided documents.

This should be tailored to Hoopla users. Hoopla is a movie streaming service.

If not enough information is available to give a good answer, say so but give as good of an answer as you can while citing the sources you have.

Query: {query}

Documents:
{docs}

Instructions:
- Provide a comprehensive answer that addresses the query
- Cite sources using [1], [2], etc. format when referencing information
- If sources disagree, mention the different viewpoints
- If the answer isn't in the documents, say "I don't have enough information"
- Be direct and informative

Answer:"""
    response = generate(model, prompt)
    print("Search Results:")
    for result in results:
        title = result.get("title", "")
        print(f"  - {title}")
    print("LLM Answer:")
    print(response.response)

def augmented_question_answering(question: str, limit: int = 5):

    hybrid_search = HybridSearch(load_movies())
    results = hybrid_search.rrf_search(question,60,limit)
    context = json.dumps([f"{result.get("title")}" for result in results])

    prompt = f"""Answer the user's question based on the provided movies that are available on Hoopla.
    
    This should be tailored to Hoopla users. Hoopla is a movie streaming service.
    
    Question: {question}
    
    Documents:
    {context}
    
    Instructions:
    - Answer questions directly and concisely
    - Be casual and conversational
    - Don't be cringe or hype-y
    - Talk like a normal person would in a chat conversation
    
    Answer:"""
    print(prompt)
    response = generate(model, prompt)
    print("Search Results:")
    for result in results:
        title = result.get("title", "")
        print(f"  - {title}")
    print("Answer:")
    print(response.response)