import os
MODEL="gemma3:1b"
from dotenv import load_dotenv
from ollama import chat
from ollama import generate
from ollama import ChatResponse

load_dotenv()

response = generate(model=MODEL,
                     prompt='Why is Boot.dev such a great place to learn about RAG? Use one paragraph maximum.')
print(response.response)
print("Prompt Tokens: 19")
print(f"Response Tokens:{response.prompt_eval_count}")
print(f"Response Tokens:{response.eval_count}")
