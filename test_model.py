from sentence_transformers import SentenceTransformer
from transformers import pipeline

print("Loading embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2")

print("Loading LLM...")
llm = pipeline("text-generation", model="google/flan-t5-small")
print("Models loaded successfully!")