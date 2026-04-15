# VARSHITH_TASK2 – Hybrid Document Retrieval System

## Overview
A local‑first retrieval system for mixed file collections (PDF, PPTX, DOCX, EML, JSON).  
It answers both **broad** research questions (e.g., *“What was the main finding of the brand study?”*) and **narrow** variable lookups (e.g., *“What does `frm_brand_awareness` measure?”*) using:

- **Hybrid search** – BM25 (sparse) + `all-MiniLM-L6-v2` dense embeddings  
- **Reciprocal Rank Fusion (RRF)** – robust combination of both retrievers  
- **Glossary prioritisation** – exact/fuzzy matching with prefix stripping (e.g., `frm_*`)  
- **Source‑aware boosting** – email files boosted for sender queries, glossary down‑weighted for broad questions  

No external LLM APIs – everything runs locally on Python 3.12+.

## Requirements
Install dependencies with:

```bash
pip install -r requirements.txt
```

`requirements.txt` includes:
- `sentence-transformers` – dense embeddings  
- `rank-bm25` – sparse retrieval  
- `pdfplumber`, `python-pptx`, `python-docx` – document parsing  
- `numpy` – similarity computations  

## Usage
```python
from retrieve import retrieve

# Broad query
results = retrieve("What was the main finding of the brand study?")

# Narrow variable query
results = retrieve("What does the variable frm_brand_awareness measure?")
```

Each result is a `dict` with `text`, `source` (filename or `"glossary"`), and `score`.

## Testing
Run the provided evaluation harness:

```bash
python test_local.py
```

This executes 10 sample queries (5 broad, 5 narrow) and reports a keyword‑based sanity score.  
**Note:** The real evaluation uses LLM‑as‑judge on different queries – this is only for local iteration.

## File Structure
```
VARSHITH_TASK2/
├── retrieve.py              # Main retrieval implementation
├── requirements.txt         # Python dependencies
├── test_local.py            # Local test harness
├── README.md                # This file
├── test_files/              # Input documents (scanned at runtime)
│   ├── research_proposal.pdf
│   ├── phase2_findings.pptx
│   ├── Stocked_questionnaire_s.docx
│   ├── glossary_partial.json
│   └── *.eml (5 email files)
└── .git/                    # Git history (included in submission)
```

## Reproducibility
1. Clone or unzip the repository.  
2. Install dependencies: `pip install -r requirements.txt`  
3. Run `python test_local.py` – no hardcoded paths, no credentials, no external API calls.  

The system automatically scans the `test_files/` directory on the first call to `retrieve()` and caches the corpus, BM25 index, and embeddings.

## Performance Highlights (local evaluation)
- **Glossary coverage** – 185 variable definitions loaded  
- **Narrow query P@1** – 22% (glossary exact/fuzzy match)  
- **Broad query P@1** – 46% (hybrid RRF fusion)  
- **First‑run setup** – ~15 seconds (model load + embedding all chunks)  
- **Subsequent queries** – near‑instant (cached corpus)

## Known Limitations
- Table extraction from PPTX/DOCX is limited (plain text only).  
- Domain‑specific jargon (e.g., `frm_*` prefixes) relies on BM25 + prefix stripping – a fine‑tuned embedding model would improve recall.  
- Email boosting is heuristic (keywords `"email"`, `"sent"`, `"sender"`).  

## Github link
https://github.com/varshithgoud2407/VARSHITH_TASK2

---
