# src/rag.py — full RAG: retrieve relevant chunks, then generate a grounded answer.

import os
from pathlib import Path
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import chromadb
from google import genai
from chunk import chunk_text
import time
from google.genai import errors
from hybrid import build_bm25, hybrid_search

def generate_with_retry(client, prompt, retries=3, wait=10):
    for attempt in range(retries):
        try:
            return client.models.generate_content(
                model="gemini-2.5-flash-lite",
                contents=prompt
            )
        except errors.ServerError as e:        # 503 — transient, retry
            if attempt < retries - 1:
                print(f"  (Gemini busy, retrying in {wait}s...)")
                time.sleep(wait)
            else:
                raise
        except errors.ClientError as e:        # 429 — quota, don't retry
            print("  (Quota exhausted — stopping, not retrying.)")
            raise
# --- Load API key from .env (never hardcode it) ---
load_dotenv()
client = genai.Client()   # auto-reads GEMINI_API_KEY from environment

# --- Rebuild the vector store (same as Day 3) ---
text = Path("data/processed/apple_10k_2025_markdown.txt").read_text(encoding="utf-8")
chunks = chunk_text(text, chunk_size=1000, overlap=150)
model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(chunks)

chroma = chromadb.PersistentClient(path="data/chroma_db")
try:
    chroma.delete_collection("apple_10k")
except Exception:
    pass
collection = chroma.create_collection("apple_10k")
collection.add(
    embeddings=[e.tolist() for e in embeddings],
    documents=chunks,
    ids=[f"chunk_{i}" for i in range(len(chunks))]
)
# Build the BM25 keyword index over the same chunks (for hybrid retrieval)
bm25 = build_bm25(chunks)
# --- The RAG function ---
def ask(question, n_results=8):
    # 1. Hybrid retrieval: fuse semantic (vector) + keyword (BM25) rankings
    retrieved = hybrid_search(question, chunks, collection, model, bm25, n_results=n_results)
    context = "\n\n---\n\n".join(retrieved)

    # 2. Build a grounded prompt — answer ONLY from context, else say so
    prompt = f"""You are a financial analyst assistant. Answer the question using ONLY the context below from Apple's 10-K filing. If the answer is not in the context, say "I don't have enough information to answer that." Do not use outside knowledge.

Context:
{context}

Question: {question}

Answer:"""

    # 3. Generate the answer
    response = generate_with_retry(client, prompt)
    return response.text, retrieved

# --- Test it ---
if __name__ == "__main__":
    question = "What was Apple's net income in 2025?"
    answer, sources = ask(question)
    print(f"Q: {question}\n")
    print(f"A: {answer}\n")