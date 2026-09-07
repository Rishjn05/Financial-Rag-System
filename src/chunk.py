from pathlib import Path

def chunk_text(text, chunk_size=1000, overlap=150):
    """Split text into chunks of `chunk_size` chars, each overlapping the
    previous by `overlap` chars so context isn't lost at the boundaries."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap
    return chunks


# This block only runs when you do `python src/chunk.py` directly —
# NOT when another file imports chunk_text. Prevents import-time side effects.
if __name__ == "__main__":
    input_path = Path("data/processed/apple_10k_2025_markdown.txt")  # updated to markdown
    text = input_path.read_text(encoding="utf-8")
    chunks = chunk_text(text, chunk_size=1000, overlap=150)
    print(f"Total characters: {len(text)}")
    print(f"Number of chunks: {len(chunks)}")
    print(f"\n--- First chunk ---\n{chunks[0]}")
    print(f"\n--- A middle chunk (#50) ---\n{chunks[50]}")