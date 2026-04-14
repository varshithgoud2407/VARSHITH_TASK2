# KEA — Task 2: Retrieval System

## What this is
Hybrid BM25 + dense retrieval system. Given a query, returns the top-5 most relevant passages across a mixed file collection (PDF, PPTX, DOCX, EML, JSON).

## How to run

```bash
pip install -r requirements.txt
python test_local.py
```

## File structure

```
yourname_task2/
├── retrieve.py          # Main implementation — retrieve() function
├── requirements.txt     # Python dependencies
├── test_local.py        # Local evaluation harness (10 sample queries)
├── README.md            # This file
├── test_files/          # Place PDF, PPTX, DOCX, EML, JSON files here
│   ├── research_proposal.pdf
│   ├── phase2_findings.pptx
│   ├── Stocked_questionnaire_s.docx
│   ├── glossary_partial.json
│   └── *.eml (5 files)
└── .git/                # Git history
```

## Approach
Hybrid retrieval: BM25 sparse + sentence-transformers dense (all-MiniLM-L6-v2).  
Scores fused with linear combination (BM25 weight boosted for narrow variable queries).  
No external LLM APIs. Runs locally on Python 3.12.

## Reproducing results
```bash
pip install -r requirements.txt
python test_local.py
```
No hardcoded paths. No credentials required.