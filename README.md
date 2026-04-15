# VARSHITH_TASK2 - Hybrid Document Retrieval System

## Overview
A local-only retrieval system for mixed file collections (PDF, PPTX, DOCX, EML, JSON).
It answers both broad research questions (for example, "What was the main finding of the brand study?") and narrow variable lookups (for example, "What does `frm_brand_awareness` measure?") using:

- Hybrid search - BM25 (sparse) + `all-MiniLM-L6-v2` dense embeddings
- Reciprocal Rank Fusion (RRF) - rank-based combination, more robust than fixed linear score weighting
- Glossary prioritisation - recursively extracts definitions from the nested `glossary_partial.json` structure, then uses exact/fuzzy matching with prefix stripping such as `frm_*`
- Source-aware boosting - email files are boosted for sender-style queries, while glossary chunks are down-weighted for broad queries

No external LLM APIs are used. Everything runs locally on Python 3.12+.

## How It Works

1. Load and chunk all documents in `test_files/` with a minimum chunk length of 50 characters.
2. Build indices:
   - BM25 over tokenised chunks
   - Dense embeddings with `all-MiniLM-L6-v2` when available
3. Extract glossary entries by recursively parsing `glossary_partial.json`.
4. Per query:
   - Detect narrow queries using variable-like patterns and definition-style phrasing.
   - If narrow, try glossary lookup first with exact, fuzzy, and prefix-stripped matching.
   - Otherwise run hybrid retrieval:
     - BM25 ranking + dense similarity
     - Source-specific boosts
     - RRF fusion
     - Return top-5 results as `text`, `source`, `score`

## Requirements

```bash
pip install -r requirements.txt
```

`requirements.txt` includes:
- `sentence-transformers` - dense embeddings
- `rank-bm25==0.2.2` - sparse retrieval
- `pdfplumber`, `python-pptx`, `python-docx` - document parsing
- `numpy` - similarity computations

## Usage

```python
from retrieve import retrieve

results = retrieve("What was the main finding of the brand study?")
results = retrieve("What does the variable frm_brand_awareness measure?")
```

Each result is a dictionary with:
- `text`
- `source`
- `score`

## Testing

```bash
python test_local.py
```

This runs 10 sample queries and reports a keyword-based sanity score.
The real evaluation uses different unseen queries and LLM-as-judge, so this script is only a local development check.

## File Structure

```text
VARSHITH_TASK2/
|-- retrieve.py
|-- requirements.txt
|-- test_local.py
|-- README.md
|-- test_files/
|   |-- research_proposal.pdf
|   |-- phase2_findings.pptx
|   |-- Stocked_questionnaire_s.docx
|   |-- glossary_partial.json
|   `-- *.eml
`-- .git/
```

## Reproducibility

1. Clone or unzip the repository.
2. Install dependencies with `pip install -r requirements.txt`.
3. Run `python test_local.py`.

The system scans `test_files/` at runtime on the first call to `retrieve()` and then caches the corpus, BM25 index, glossary, and dense embeddings in memory.

## Expected Performance

Broad queries and narrow glossary lookups are both supported, but the exact score depends on the local environment and whether dense embeddings are available.
The provided `test_local.py` output is a sanity check only and should not be treated as the official evaluation score.

## GitHub Repository

https://github.com/varshithgoud2407/VARSHITH_TASK2
