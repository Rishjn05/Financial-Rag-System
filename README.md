 # 10-K Financial RAG — Question Answering over SEC Annual Reports

A retrieval-augmented generation (RAG) system that answers natural-language questions about a company's 10-K annual report, grounded in the filing itself. Built end-to-end in Python — document ingestion, chunking, embeddings, vector search, keyword search, and answer generation — with a custom evaluation harness used to **measure failures and drive each improvement**.

The project's focus is not "a chatbot that works," but an honest, measured engineering loop: build a baseline, evaluate it, diagnose where and *why* it fails, fix the root cause, and re-measure.

---

## What it does

Ask a question like *"What was Apple's net income in 2025?"* and the system retrieves the most relevant passages from the 10-K and generates a grounded answer that cites only the filing — refusing to answer when the information isn't present.

**Example:**
> **Q:** What is the total number of employees in Google?
> **A:** I don't have enough information to answer that.

(The system correctly refuses, because that fact is not in Apple's filing — grounding, not hallucination.)

---

## Architecture

The pipeline is built from independent, swappable components:

```
SEC 10-K (HTML)
   │  extract  (PyMuPDF / pymupdf4llm → Markdown, tables preserved)
   ▼
Plain / Markdown text
   │  chunk    (fixed-size, 1000 chars, 150 overlap)
   ▼
Text chunks
   │  embed    (sentence-transformers, all-MiniLM-L6-v2 → 384-dim vectors)
   ▼
Vector store (ChromaDB)  +  Keyword index (BM25)
   │  retrieve (hybrid: semantic + keyword, fused with Reciprocal Rank Fusion)
   ▼
Top-k relevant chunks
   │  generate (Gemini, grounded prompt: "answer ONLY from context")
   ▼
Grounded answer
```

**Key design choices and why:**

- **RAG, not fine-tuning** — 10-K data is factual, changes yearly, and must be verifiable. RAG lets new filings be added instantly, cites the source chunk, and grounds answers to reduce hallucination. Fine-tuning bakes facts into weights (costly to update, no source attribution).
- **Hand-rolled pipeline (no LangChain)** — every stage is explicit, so each component's behavior is understood and defensible rather than hidden behind a framework.
- **`all-MiniLM-L6-v2` embeddings** — small, fast, standard baseline; the tradeoff is lower richness than larger models.
- **Hybrid retrieval (semantic + BM25)** — semantic search captures meaning; BM25 captures exact terms. Together they cover each other's blind spots.
- **Grounded prompt** — the model is instructed to answer only from retrieved context and to refuse otherwise, which is the primary defense against hallucination.

---

## Evaluation

Rather than eyeballing answers, the project uses a **categorized test set** with automated scoring across five question types:

| Category | Tests | Example |
|---|---|---|
| Prose list | retrieval of list-style facts | product announcements by quarter |
| Nested table (multi-part) | hard table extraction | Japan net sales + % change |
| Not present | hallucination / refusal | "employees in Google?" |
| Prose reasoning | qualitative answers | why iPhone sales rose |
| Table comparison | reasoning across rows | highest / lowest product category |
| Arithmetic | computed answers | basic vs diluted EPS difference |
| Ambiguous phrasing | retrieval robustness | "how profitable was Apple?" |

Two metrics: **retrieval recall** (was the correct chunk retrieved?) and **answer correctness** (did the final answer contain the ground-truth value, or correctly refuse?).

**Headline results (on `gemini-2.5-flash`):** ~71% (5 / 7) on the hard set.

**Consistently passing:** refusal/hallucination test, prose reasoning, arithmetic.
**Consistently failing:** the nested segment-table query (Japan) and an ambiguous-phrasing query.

---

## The interesting part: a failure traced to its root cause

The Japan net-sales question failed across **every** version of the pipeline, and chasing it down was the most instructive part of the project. The failure *evolved* with each fix, which is what pinpointed the true cause:

1. **Naive text extraction** → financial tables were flattened from 2-D into a 1-D stream (`get_text()` reads top-to-bottom, destroying column structure). The model returned a *different wrong number on each run* — non-determinism that proved it had no reliable signal and was guessing.
2. **Structure-aware parsing (`pymupdf4llm` → Markdown)** → tables preserved as Markdown, but cells were polluted with `<br>` tags and the header row was conflated with the first data row. The model now gave a *stable but wrong* value.
3. **`<br>` cleanup** → cleaner cells, but the answer became a refusal — the table chunk was no longer reliably retrieved.
4. **Hybrid retrieval (BM25 + semantic)** → still didn't surface the segment table; instead, *which* fragment surfaced changed, sometimes breaking a previously-passing question.

**Root cause:** the fixed-size chunker **splits large tables across chunk boundaries**. Once a table is fragmented, no retrieval strategy can reliably reassemble it — so failures move around rather than resolve. The correct fix is **table-aware chunking** (treating each table as an atomic unit), which is the documented next step.

This is the key engineering lesson of the project: **the bottleneck was chunking, not parsing or retrieval** — and only systematic evaluation across multiple fixes made that visible.

---

## Honest limitations

- **Large tables split across chunks** — the primary unsolved failure (above). Fix: table-aware chunking.
- **Evaluation is model-sensitive** — the same pipeline scored ~71% on `gemini-2.5-flash` vs ~57% on the weaker `gemini-2.5-flash-lite`. RAG evaluation must hold the generator model constant; comparing across models confounds results.
- **Small test set (7 questions)** — a baseline, not a comprehensive benchmark.
- **Naive BM25 tokenization** (whitespace split) — production would use a proper tokenizer.
- **OCR invoked during parsing** — can introduce text errors on image-based elements.
- **Free-tier API quotas** — daily limits constrained eval iteration; production would use a paid tier or cache results.

---

## Tech stack

Python · PyMuPDF / pymupdf4llm · sentence-transformers · ChromaDB · rank-bm25 · Google Gemini API · python-dotenv

---

## Project structure

```
src/
  extract.py            # naive text extraction (baseline)
  extract_markdown.py   # structure-aware extraction (pymupdf4llm)
  chunk.py              # fixed-size overlapping chunker
  embed.py              # embedding step (standalone checkpoint)
  retrieve.py           # vector store + retrieval
  hybrid.py             # BM25 + semantic fusion (RRF)
  rag.py                # full pipeline: retrieve → grounded generation
  evaluate.py           # retrieval recall metric
  evaluate_faithfulness.py  # answer-correctness metric
  evaluate_hard.py      # categorized hard test set
```

---


