# src/evaluate.py — measure retrieval recall on a small test set.

from pathlib import Path
from sentence_transformers import SentenceTransformer
import chromadb
from chunk import chunk_text

# --- Rebuild the store (same pipeline) ---
text = Path("data/processed/apple_10k_2025_markdown.txt").read_text(encoding="utf-8")
chunks = chunk_text(text, chunk_size=1000, overlap=150)
model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(chunks)

client = chromadb.PersistentClient(path="data/chroma_db")
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

# --- Test set: question + a string that MUST appear in the correct chunk ---
# The "must_contain" string is how we identify the right chunk automatically.
test_set = [
    {"question": "What was Apple's net income?",            "must_contain": "112,010"},
    {"question": "What was Apple's net income in 2025?",    "must_contain": "112,010"},
    {"question": "What were Apple's total net sales?",      "must_contain": "416,161"},
    {"question": "What were Apple's net sales in China?",   "must_contain": "64,377"},
    # add 4-6 more of your own using real numbers from the filing
]

# --- Measure recall@N ---
def evaluate(n_results=3):
    hits = 0
    for item in test_set:
        q_emb = model.encode([item["question"]])
        results = collection.query(query_embeddings=q_emb.tolist(), n_results=n_results)
        retrieved = results["documents"][0]
        # Did any retrieved chunk contain the ground-truth string?
        found = any(item["must_contain"] in chunk for chunk in retrieved)
        hits += found
        print(f"{'✓' if found else '✗'}  {item['question']}")
    recall = hits / len(test_set)
    print(f"\nRecall@{n_results}: {recall:.0%}  ({hits}/{len(test_set)})")

if __name__ == "__main__":
    evaluate(n_results=8)