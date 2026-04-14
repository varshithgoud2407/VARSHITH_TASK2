"""
retrieve.py — Hybrid BM25 + Dense Retrieval System
Professional implementation targeting KEA Task 2.
"""

import os
import json
import math
import re
from pathlib import Path
from typing import List, Dict

# ── OPTIONAL IMPORTS (graceful fallback) ─────────────────────────────────────
try:
    from sentence_transformers import SentenceTransformer
    import numpy as np
    DENSE_AVAILABLE = True
except ImportError:
    DENSE_AVAILABLE = False

try:
    import pdfplumber
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False

try:
    from pptx import Presentation
    PPTX_AVAILABLE = True
except ImportError:
    PPTX_AVAILABLE = False

try:
    from docx import Document
    DOCX_AVAILABLE = True
except ImportError:
    DOCX_AVAILABLE = False

try:
    import email
    from email import policy
    EMAIL_AVAILABLE = True
except ImportError:
    EMAIL_AVAILABLE = False


# ── CONSTANTS ─────────────────────────────────────────────────────────────────
TEST_FILES_DIR = Path(__file__).parent / "test_files"
CHUNK_SIZE = 200 
CHUNK_OVERLAP = 40
TOP_K = 5
BM25_WEIGHT = 0.35
DENSE_WEIGHT = 0.65
MODEL_NAME = "all-MiniLM-L6-v2"


# ── TEXT EXTRACTION ────────────────────────────────────────────────────────────

def extract_pdf(path: Path) -> List[Dict]:
    chunks = []
    if not PDF_AVAILABLE: return chunks
    try:
        with pdfplumber.open(path) as pdf:
            for page_num, page in enumerate(pdf.pages, 1):
                text = page.extract_text() or ""
                # Table extraction helps with structured survey data
                tables = page.extract_tables()
                for table in tables:
                    if table:
                        for row in table:
                            row_text = " | ".join(str(c) for c in row if c)
                            text += "\n" + row_text
                for chunk in _chunk_text(text.strip()):
                    chunks.append({"text": chunk, "source": path.name, "meta": f"page {page_num}"})
    except Exception: pass
    return chunks

def extract_pptx(path: Path) -> List[Dict]:
    chunks = []
    if not PPTX_AVAILABLE: return chunks
    try:
        prs = Presentation(path)
        for slide_num, slide in enumerate(prs.slides, 1):
            texts = [shape.text.strip() for shape in slide.shapes if hasattr(shape, "text") and shape.text.strip()]
            full_text = "\n".join(texts)
            for chunk in _chunk_text(full_text):
                chunks.append({"text": chunk, "source": path.name, "meta": f"slide {slide_num}"})
    except Exception: pass
    return chunks

def extract_docx(path: Path) -> List[Dict]:
    chunks = []
    if not DOCX_AVAILABLE: return chunks
    try:
        doc = Document(path)
        full_text = "\n".join([para.text.strip() for para in doc.paragraphs if para.text.strip()])
        for chunk in _chunk_text(full_text):
            chunks.append({"text": chunk, "source": path.name, "meta": "document"})
    except Exception: pass
    return chunks

def extract_eml(path: Path) -> List[Dict]:
    """FIXED: Uses message_from_bytes for cross-platform reliability."""
    chunks = []
    if not EMAIL_AVAILABLE: return chunks
    try:
        with open(path, "rb") as f:
            # Using from_bytes to avoid Windows policy.default behavior bugs
            msg = email.message_from_bytes(f.read(), policy=policy.default)
        
        subject = msg.get("Subject", "")
        sender = msg.get("From", "")
        body = msg.get_content() if not msg.is_multipart() else ""
        if msg.is_multipart():
            for part in msg.walk():
                if part.get_content_type() == "text/plain":
                    body += part.get_content() or ""
        
        full_text = f"Subject: {subject}\nFrom: {sender}\n{body}"
        cleaned = re.sub(r'\s+', ' ', re.sub(r'<[^>]+>', ' ', full_text)).strip()
        
        for chunk in _chunk_text(cleaned):
            chunks.append({"text": chunk, "source": path.name, "meta": "email"})
    except Exception: pass
    return chunks

def extract_json_glossary(path: Path) -> List[Dict]:
    """FIXED: Correctly targets the 'V' schema for variables."""
    chunks = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        # Target the 'V' key containing the 180+ variable definitions
        variables = data.get("V", {})
        for var_name, info in variables.items():
            definition = info.get("definition", "")
            # Aliasing: ensure both name and definition are indexed for retrieval 
            rich_text = f"Variable Name: {var_name}\nDefinition: {definition}"
            chunks.append({
                "text": rich_text,
                "source": "glossary",
                "meta": var_name
            })
    except Exception: pass
    return chunks

def _chunk_text(text: str) -> List[str]:
    words = text.split()
    return [" ".join(words[i:i + CHUNK_SIZE]) for i in range(0, len(words), CHUNK_SIZE - CHUNK_OVERLAP)]


# ── RETRIEVAL LOGIC ───────────────────────────────────────────────────────────

class BM25:
    def __init__(self, corpus: List[str]):
        self.corpus = corpus
        self.tokenized = [re.findall(r'\b\w+\b', d.lower()) for d in corpus]
        self.n = len(corpus)
        self.avgdl = sum(len(d) for d in self.tokenized) / max(self.n, 1)
        self.idf = self._compute_idf()

    def _compute_idf(self):
        df = {}
        for doc in self.tokenized:
            for term in set(doc): df[term] = df.get(term, 0) + 1
        return {t: math.log((self.n - f + 0.5) / (f + 0.5) + 1) for t, f in df.items()}

    def score(self, query: str) -> List[float]:
        q_terms = re.findall(r'\b\w+\b', query.lower())
        scores = []
        for doc in self.tokenized:
            s, dl = 0.0, len(doc)
            for t in q_terms:
                if t in self.idf:
                    tf = doc.count(t)
                    s += self.idf[t] * (tf * 2.5) / (tf + 1.5 * (0.25 + 0.75 * dl / self.avgdl))
            scores.append(s)
        return scores

_model, _embeddings, _chunks_cache = None, None, None

def retrieve(query: str) -> List[Dict]:
    global _embeddings, _chunks_cache, _model
    if _chunks_cache is None:
        _chunks_cache = []
        extractors = {"*.pdf": extract_pdf, "*.pptx": extract_pptx, "*.docx": extract_docx, "*.eml": extract_eml, "*.json": extract_json_glossary}
        for pattern, func in extractors.items():
            for path in TEST_FILES_DIR.glob(pattern):
                _chunks_cache.extend(func(Path(path)))

    if not _chunks_cache: return []
    texts = [c["text"] for c in _chunks_cache]

    # BM25 Sparse [cite: 397]
    bm25_scores = BM25(texts).score(query)
    bm25_norm = [(s - min(bm25_scores)) / (max(bm25_scores) - min(bm25_scores) + 1e-9) for s in bm25_scores]

    # Dense Semantic [cite: 397, 423]
    dense_norm = [0.0] * len(texts)
    if DENSE_AVAILABLE:
        if _model is None: _model = SentenceTransformer(MODEL_NAME)
        if _embeddings is None: _embeddings = _model.encode(texts, convert_to_numpy=True)
        q_emb = _model.encode([query])[0]
        sims = (_embeddings @ q_emb) / (np.linalg.norm(_embeddings, axis=1) * np.linalg.norm(q_emb) + 1e-9)
        dense_norm = [(s - min(sims)) / (max(sims) - min(sims) + 1e-9) for s in sims]

    # Hybrid Fusion [cite: 397, 412]
    # Heuristic: Boost BM25 if underscores are present (likely variable lookup) 
    bw = 0.7 if "_" in query else BM25_WEIGHT
    dw = 1.0 - bw
    fused = [bw * b + dw * d for b, d in zip(bm25_norm, dense_norm)]

    ranked = sorted(range(len(fused)), key=lambda i: fused[i], reverse=True)
    results, seen = [], set()
    for idx in ranked:
        if len(results) >= TOP_K: break
        txt = _chunks_cache[idx]["text"]
        if txt[:50] not in seen:
            seen.add(txt[:50])
            results.append({"text": txt, "source": _chunks_cache[idx]["source"], "score": round(fused[idx], 4)})
    return results

if __name__ == "__main__":
    # Sanity check
    for r in retrieve("What does frm_brand_awareness measure?"):
        print(f"[{r['score']}] ({r['source']}) {r['text'][:100]}...")