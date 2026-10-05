# DocuMind AI

A fully local, privacy-preserving Retrieval-Augmented Generation (RAG) system for question-answering over your own PDF documents. Runs entirely on your machine via [Ollama](https://ollama.com) — no external API calls, no data leaves your device.

## Features

- **Hybrid retrieval**: combines dense vector search (FAISS) with lexical search (BM25), fused via Reciprocal Rank Fusion (RRF)
- **Cross-encoder reranking** for higher-precision top-K results
- **Grounded generation**: answers are constrained to retrieved context only, with inline source + page citations, reducing hallucination
- **Interactive Streamlit chat UI** with live ingestion progress, retrieval/generation timing, and expandable source evidence
- **Retrieval evaluation harness**: Hit@1, Hit@3, MRR, and latency, measured against a labeled question set

## Architecture

```
PDF Upload
    ↓
Text Extraction (page-level, pypdf)
    ↓
Chunking (overlapping word-based windows)
    ↓
Embedding Generation (Ollama — embeddinggemma)
    ↓
Index Building (FAISS dense index + BM25 lexical index)
    ↓
─────────────── Query time ───────────────
    ↓
Dense Search (FAISS) + BM25 Search  → Reciprocal Rank Fusion
    ↓
Cross-Encoder Reranking (MiniLM-L-6-v2)
    ↓
Grounded Answer Generation (Ollama — Llama 3.2), with citations
```

## Tech Stack

| Component | Technology |
|---|---|
| Dense retrieval | FAISS (`IndexFlatIP`, cosine similarity via L2-normalized embeddings) |
| Lexical retrieval | BM25 (`rank_bm25`) |
| Fusion | Reciprocal Rank Fusion (RRF) |
| Reranking | Cross-encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`) |
| Embeddings | Ollama (`embeddinggemma`) |
| Generation | Ollama (`llama3.2:3b`) |
| PDF parsing | `pypdf` |
| UI | Streamlit |

## Project Structure

```
project_root/
├── app.py                     # Streamlit app entry point
├── src/localrag/
│   ├── config.py               # Paths, model names, retrieval hyperparameters
│   ├── document_loader.py      # PDF/TXT/MD text extraction
│   ├── chunker.py               # Overlapping chunking
│   ├── embeddings.py           # Ollama embedding calls
│   ├── vector_store.py         # FAISS index build/load/search
│   ├── bm25_store.py            # BM25 index build/load/search
│   ├── retriever.py            # RRF fusion + reranking orchestration
│   ├── reranker.py              # Cross-encoder reranking
│   ├── generator.py            # Grounded answer generation
│   └── ingestion.py            # End-to-end ingestion pipeline
├── scripts/
│   └── evaluate.py             # Retrieval evaluation harness
├── evaluation/
│   └── questions.json          # Labeled question set for evaluation
└── data/
    ├── documents/               # Uploaded PDFs + papers.json metadata
    └── index/                   # faiss.index, chunks.json, bm25.pkl
```

## Setup

### 1. Install dependencies

```bash
pip install streamlit faiss-cpu rank_bm25 sentence-transformers pypdf ollama numpy
```

### 2. Install and start Ollama

Download from [ollama.com](https://ollama.com), then pull the required models:

```bash
ollama pull embeddinggemma
ollama pull llama3.2:3b
```

### 3. Run the app

```bash
streamlit run app.py
```

Upload PDFs through the UI and click **Build Corpus** to run ingestion — this writes `faiss.index`, `chunks.json`, and `bm25.pkl` to `data/index/`.

## Retrieval Evaluation

The evaluation harness (`scripts/evaluate.py`) measures retrieval quality against a labeled question set at `evaluation/questions.json`. Each entry specifies a question and its `relevant_sources` (the filename(s) that should be retrieved).

Run it after building a corpus:

```bash
python scripts/evaluate.py
```

It reports per-question and aggregate **Hit@1**, **Hit@3**, **Mean Reciprocal Rank (MRR)**, and retrieval latency.

### Latest results

| Corpus | Questions | Hit@1 | Hit@3 | MRR |
|---|---|---|---|---|
| 9 documents (197 chunks) | 83 | 72.3% | 83.1% | 0.775 |

**Notes on interpreting these numbers:**
- The corpus spans multiple related but distinct topics (India-China border relations, India's broader foreign policy history, India-Russia relations, India-US relations), giving retrieval genuine documents to discriminate between rather than a single-topic corpus with little competition.
- The question set includes deliberately hard cases: paraphrased questions that avoid the source's exact terminology, and multi-document questions where more than one source is a valid answer.
- These numbers should be read as validation that the retrieval pipeline works and can meaningfully discriminate between related documents — not as a general-purpose benchmark of ranking quality on arbitrary or much larger corpora.

To add more documents or questions and rerun the evaluation, see `evaluation/questions.json` for the expected format:

```json
[
  {
    "question": "Your question here",
    "relevant_sources": ["exact_filename_as_indexed.pdf"]
  }
]
```

`relevant_sources` values must exactly match the filenames as ingested (check the "Indexed documents:" line printed at the top of `evaluate.py`'s output if a question isn't matching as expected).

## Limitations

- Corpus is fully rebuilt on each ingestion (`clear_previous_corpus()`) — not incremental.
- PDF-only ingestion is wired to the upload pipeline; `.txt`/`.md` support exists in `document_loader.py` but isn't exposed in the Streamlit UI.
- Retrieval and generation run sequentially on CPU by default, dependent on local Ollama performance.
