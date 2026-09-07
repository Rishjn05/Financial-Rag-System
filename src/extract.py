# src/extract.py
# Downloads a 10-K HTML document from a URL and extracts text to a .txt file.

import pymupdf
import urllib.request
from pathlib import Path

# --- The SEC document URL (paste yours here) ---
url = "https://www.sec.gov/Archives/edgar/data/320193/000032019325000079/aapl-20250927.htm"
output_path = Path("data/processed/apple_10k_2025.txt")

# --- SEC requires a User-Agent header or it blocks the request ---
# Use your name + email; SEC asks filers to identify themselves.
req = urllib.request.Request(url, headers={"User-Agent": "Rashika rashika112005@gmail.com"})
html_bytes = urllib.request.urlopen(req).read()

# --- Open the downloaded HTML with PyMuPDF (filetype tells it it's HTML) ---
doc = pymupdf.open(stream=html_bytes, filetype="html")
print(f"Opened document — {doc.page_count} pages")

# --- Extract text from every page ---
all_text = []
for page in doc:
    all_text.append(page.get_text())

full_text = "\n\n----- PAGE BREAK -----\n\n".join(all_text)
output_path.write_text(full_text, encoding="utf-8")
print(f"Done. Saved {len(full_text)} characters to {output_path}")
doc.close()