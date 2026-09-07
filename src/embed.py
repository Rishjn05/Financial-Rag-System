from pathlib import Path
from sentence_transformers import SentenceTransformer
from chunk import chunk_text

text=Path("data/processed/apple_10k_2025.txt").read_text(encoding="utf-8")
chunks=chunk_text(text , chunk_size=1000 , overlap=150)
print(f"Chunks to embed: {len(chunks)}")

model=SentenceTransformer("all-MiniLM-L6-v2")
embeddings=model.encode(chunks , show_progress_bar=True)

# --- Confirm it worked ---
print(f"Embeddings shape: {embeddings.shape}")   # (num_chunks, 384)
print(f"First vector (first 10 numbers): {embeddings[0][:10]}")
