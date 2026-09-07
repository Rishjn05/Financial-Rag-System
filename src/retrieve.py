from pathlib import Path
from sentence_transformers import SentenceTransformer
import chromadb
from chunk import chunk_text

# --- 1. Load, chunk, embed (same pipeline as before) ---
text = Path("data/processed/apple_10k_2025.txt").read_text(encoding="utf-8")
chunks = chunk_text(text, chunk_size=1000, overlap=150)
model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(chunks, show_progress_bar=True)
print(f"Embedded {len(chunks)} chunks")

# --- 2. Set up ChromaDB (persistent: saved to disk so you don't re-embed every run) ---
client=chromadb.PersistentClient(path="data/chroma_db")
try:
    client.delete_collection("apple_10k")
except Exception:
    pass
collection = client.create_collection("apple_10k")

collection.add(
    embeddings=[e.tolist() for e in embeddings],
    documents=chunks,
    ids=[f"chunk_{i}" for i in range(len(chunks))]
)
print(f"Stored {collection.count()} chunks in ChromaDB")
# --- 4. Retrieve: ask a question, get the nearest chunks ---
question = "What was Apple's net income?"
q_embedding = model.encode([question])            # embed the QUESTION (same model!)
results = collection.query(
    query_embeddings=q_embedding.tolist(),
    n_results=3                                    # top 3 nearest chunks
)
# --- 5. Show what came back ---
print(f"\nQuestion: {question}\n")
for i, doc in enumerate(results["documents"][0]):
    print(f"--- Retrieved chunk {i+1} ---")
    print(doc[:400])   # first 400 chars of each
    print()