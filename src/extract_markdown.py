# src/extract_markdown.py — structure-aware extraction (preserves tables as Markdown).

import pymupdf4llm
import urllib.request
from pathlib import Path
import re

url = "https://www.sec.gov/Archives/edgar/data/320193/000032019325000079/aapl-20250927.htm"
output_path = Path("data/processed/apple_10k_2025_markdown.txt")  # NEW file, keeps baseline

# Download the HTML (same as before — SEC needs a User-Agent)
req = urllib.request.Request(url, headers={"User-Agent": "Rashika rashika112005@gmail.com"})
html_bytes = urllib.request.urlopen(req).read()

# Save to a temp file because pymupdf4llm reads from a file path
tmp = Path("data/raw/apple_10k_2025.htm")
tmp.write_bytes(html_bytes)

# THE KEY LINE: convert to Markdown, preserving table structure
md_text = pymupdf4llm.to_markdown(tmp)

# --- Clean up: remove <br> tags that pollute every table cell ---
# pymupdf4llm leaves HTML <br> tags inside cells, which clutter the text
# and confuse the model when reading tables. Replace with a space.
md_text = md_text.replace("<br>", " ")
# Collapse any doubled spaces the replacement may have created.

md_text = re.sub(r" {2,}", " ", md_text)

output_path.write_text(md_text, encoding="utf-8")
print(f"Done. Saved {len(md_text)} characters of Markdown to {output_path}")