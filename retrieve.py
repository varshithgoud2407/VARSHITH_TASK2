"""
retrieve.py — Hybrid BM25 + Dense Retrieval System
Handles broad narrative queries and narrow variable lookups across
PDF, PPTX, DOCX, EML, and JSON files in test_files/.

Usage:
    from retrieve import retrieve
    results = retrieve("What was the main finding of the brand study?")
"""

import os
import json
import math
import re
import glob
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
CHUNK_SIZE = 200        # tokens (approx words)
CHUNK_OVERLAP = 40
TOP_K = 5
BM25_WEIGHT = 0.35
DENSE_WEIGHT = 0.65
MODEL_NAME = "all-MiniLM-L6-v2"


# ── TEXT EXTRACTION ────────────────────────────────────────────────────────────

def extract_pdf(path: Path) -> List[Dict]:
    """Extract text chunks from a PDF file."""
    chunks = []
    if not PDF_AVAILABLE:
        return chunks
    try:
        with pdfplumber.open(path) as pdf:
            for page_num, page in enumerate(pdf.pages, 1):
                text = page.extract_text() or ""
                # Also try extracting tables as text
                tables = page.extract_tables()
                for table in tables:
                    if table:
                        for row in table:
                            row_text = " | ".join(str(c) for c in row if c)
                            text += "\n" + row_text
                for chunk in _chunk_text(text.strip()):
                    chunks.append({
                        "text": chunk,
                        "source": path.name,
                        "meta": f"page {page_num}"
                    })
    except Exception as e:
        pass
    return chunks


def extract_pptx(path: Path) -> List[Dict]:
    """Extract text from PowerPoint slides."""
    chunks = []
    if not PPTX_AVAILABLE:
        return chunks
    try:
        prs = Presentation(path)
        for slide_num, slide in enumerate(prs.slides, 1):
            texts = []
            for shape in slide.shapes:
                if hasattr(shape, "text") and shape.text.strip():
                    texts.append(shape.text.strip())
                # Extract table cells
                if shape.has_table:
                    for row in shape.table.rows:
                        row_text = " | ".join(
                            cell.text.strip() for cell in row.cells if cell.text.strip()
                        )
                        if row_text:
                            texts.append(row_text)
            full_text = "\n".join(texts)
            for chunk in _chunk_text(full_text):
                chunks.append({
                    "text": chunk,
                    "source": path.name,
                    "meta": f"slide {slide_num}"
                })
    except Exception:
        pass
    return chunks


def extract_docx(path: Path) -> List[Dict]:
    """Extract text from Word documents."""
    chunks = []
    if not DOCX_AVAILABLE:
        return chunks
    try:
        doc = Document(path)
        texts = []
        for para in doc.paragraphs:
            if para.text.strip():
                texts.append(para.text.strip())
        # Also extract tables
        for table in doc.tables:
            for row in table.rows:
                row_text = " | ".join(
                    cell.text.strip() for cell in row.cells if cell.text.strip()
                )
                if row_text:
                    texts.append(row_text)
        full_text = "\n".join(texts)
        for chunk in _chunk_text(full_text):
            chunks.append({
                "text": chunk,
                "source": path.name,
                "meta": "document"
            })
    except Exception:
        pass
    return chunks


def extract_eml(path: Path) -> List[Dict]:
    """Extract text from email files."""
    chunks = []
    if not EMAIL_AVAILABLE:
        return chunks
    try:
        with open(path, "rb") as f:
            msg = email.message_from_binary_file(f, policy=policy.default)
        
        subject = msg.get("Subject", "")
        sender = msg.get("From", "")
        date = msg.get("Date", "")
        
        body = ""
        if msg.is_multipart():
            for part in msg.walk():
                ctype = part.get_content_type()
                if ctype == "text/plain":
                    body += part.get_content() or ""
        else:
            body = msg.get_content() or ""
        
        # Clean HTML if present
        body = re.sub(r'<[^>]+>', ' ', body)
        body = re.sub(r'\s+', ' ', body).strip()
        
        header = f"Subject: {subject} | From: {sender} | Date: {date}\n"
        full_text = header + body
        
        for chunk in _chunk_text(full_text):
            chunks.append({
                "text": chunk,
                "source": path.name,
                "meta": f"email from {sender[:30]}"
            })
    except Exception:
        pass
    return chunks


def extract_json_glossary(path: Path) -> List[Dict]:
    """Extract variable definitions from glossary JSON."""
    chunks = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        # Handle both list and dict structures
        entries = []
        if isinstance(data, list):
            entries = data
        elif isinstance(data, dict):
            # Could be {variable_name: definition} or {variables: [...]}
            if "variables" in data:
                entries = data["variables"]
            else:
                entries = [{"name": k, **v} if isinstance(v, dict) else {"name": k, "description": str(v)}
                          for k, v in data.items()]
        
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            # Build a rich text representation of the variable
            name = entry.get("name", entry.get("variable", entry.get("var_name", "")))
            description = entry.get("description", entry.get("label", entry.get("text", "")))
            scale = entry.get("scale", entry.get("values", ""))
            
            if name or description:
                text = f"Variable: {name}\nDescription: {description}"
                if scale:
                    text += f"\nScale/Values: {scale}"
                chunks.append({
                    "text": text,
                    "source": "glossary",
                    "meta": f"variable: {name}"
                })
    except Exception:
        pass
    return chunks


def _chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """Split text into overlapping word chunks."""
    if not text or not text.strip():
        return []
    words = text.split()
    if not words:
        return []
    chunks = []
    for i in range(0, len(words), size - overlap):
        chunk = " ".join(words[i:i + size])
        if chunk.strip():
            chunks.append(chunk)
    return chunks


# ── DOCUMENT LOADING ──────────────────────────────────────────────────────────

def load_all_documents(base_dir: Path = TEST_FILES_DIR) -> List[Dict]:
    """Scan test_files/ at runtime and extract all chunks. No hardcoded filenames."""
    all_chunks = []
    
    if not base_dir.exists():
        return all_chunks
    
    extractors = {
        "*.pdf": extract_pdf,
        "*.pptx": extract_pptx,
        "*.docx": extract_docx,
        "*.eml": extract_eml,
    }
    
    for pattern, extractor in extractors.items():
        for path in base_dir.glob(pattern):
            chunks = extractor(Path(path))
            all_chunks.extend(chunks)
    
    # Handle JSON glossary files
    for path in base_dir.glob("*.json"):
        chunks = extract_json_glossary(Path(path))
        all_chunks.extend(chunks)
    
    return all_chunks


# ── BM25 IMPLEMENTATION ───────────────────────────────────────────────────────

class BM25:
    """Lightweight BM25 implementation — no external dependency."""
    
    def __init__(self, corpus: List[str], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus = corpus
        self.tokenized = [self._tokenize(doc) for doc in corpus]
        self.n = len(corpus)
        self.avgdl = sum(len(d) for d in self.tokenized) / max(self.n, 1)
        self.idf = self._compute_idf()
    
    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r'\b\w+\b', text.lower())
    
    def _compute_idf(self) -> Dict[str, float]:
        df: Dict[str, int] = {}
        for doc in self.tokenized:
            for term in set(doc):
                df[term] = df.get(term, 0) + 1
        idf = {}
        for term, freq in df.items():
            idf[term] = math.log((self.n - freq + 0.5) / (freq + 0.5) + 1)
        return idf
    
    def score(self, query: str) -> List[float]:
        q_terms = self._tokenize(query)
        scores = []
        for doc_tokens in self.tokenized:
            score = 0.0
            dl = len(doc_tokens)
            tf_counter: Dict[str, int] = {}
            for t in doc_tokens:
                tf_counter[t] = tf_counter.get(t, 0) + 1
            for term in q_terms:
                if term not in self.idf:
                    continue
                tf = tf_counter.get(term, 0)
                idf_val = self.idf[term]
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
                score += idf_val * (numerator / denominator)
            scores.append(score)
        return scores


# ── DENSE RETRIEVAL ───────────────────────────────────────────────────────────

_model = None
_embeddings = None
_chunks_cache = None


def _get_model():
    global _model
    if _model is None and DENSE_AVAILABLE:
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def _normalise(scores: List[float]) -> List[float]:
    """Min-max normalise a list of scores to [0, 1]."""
    if not scores:
        return scores
    mn, mx = min(scores), max(scores)
    if mx == mn:
        return [0.0] * len(scores)
    return [(s - mn) / (mx - mn) for s in scores]


# ── QUERY TYPE DETECTION ──────────────────────────────────────────────────────

def _is_narrow_variable_query(query: str) -> bool:
    """Detect narrow variable lookup queries (e.g. 'what does frm_brand_awareness measure?')."""
    # Contains underscore-separated token that looks like a variable name
    if re.search(r'\b[a-z]{2,}_[a-z]', query.lower()):
        return True
    # Asks about a specific variable / measure / metric by coded name
    if re.search(r'\b(variable|measure|metric|column|field)\b', query.lower()):
        return True
    return False


# ── MAIN RETRIEVE FUNCTION ────────────────────────────────────────────────────

def retrieve(query: str) -> List[Dict]:
    """
    Return top-5 most relevant passages for the given query.
    
    Each result is a dict with:
        - "text": str    — the passage content
        - "source": str  — source filename or "glossary"
        - "score": float — relevance score (higher = better)
    
    Handles two query types:
        - Broad:  "What was the main finding of the brand study?"
        - Narrow: "What does the variable frm_brand_awareness measure?"
    """
    global _embeddings, _chunks_cache
    
    # Load and index documents (cached after first call)
    if _chunks_cache is None:
        _chunks_cache = load_all_documents()
    
    chunks = _chunks_cache
    
    if not chunks:
        return []
    
    texts = [c["text"] for c in chunks]
    
    # ── BM25 SCORES ───────────────────────────────────────────────────────────
    bm25 = BM25(texts)
    bm25_raw = bm25.score(query)
    bm25_norm = _normalise(bm25_raw)
    
    # ── DENSE SCORES ──────────────────────────────────────────────────────────
    dense_norm = [0.0] * len(texts)
    if DENSE_AVAILABLE:
        model = _get_model()
        if _embeddings is None:
            _embeddings = model.encode(texts, batch_size=64, show_progress_bar=False)
        q_emb = model.encode([query], show_progress_bar=False)[0]
        # Cosine similarity
        norms = np.linalg.norm(_embeddings, axis=1)
        q_norm = np.linalg.norm(q_emb)
        if q_norm > 0:
            sims = (_embeddings @ q_emb) / (norms * q_norm + 1e-9)
            dense_norm = _normalise(sims.tolist())
    
    # ── HYBRID FUSION ─────────────────────────────────────────────────────────
    # For narrow variable queries, boost BM25 weight
    is_narrow = _is_narrow_variable_query(query)
    bw = 0.6 if is_narrow else BM25_WEIGHT
    dw = 0.4 if is_narrow else DENSE_WEIGHT
    
    fused = [
        bw * b + dw * d
        for b, d in zip(bm25_norm, dense_norm)
    ]
    
    # ── RANK AND DEDUPLICATE ──────────────────────────────────────────────────
    ranked_indices = sorted(range(len(fused)), key=lambda i: fused[i], reverse=True)
    
    results = []
    seen_texts = set()
    
    for idx in ranked_indices:
        if len(results) >= TOP_K:
            break
        
        chunk = chunks[idx]
        # Simple deduplication — skip near-identical passages
        text_key = chunk["text"][:100].lower().strip()
        if text_key in seen_texts:
            continue
        seen_texts.add(text_key)
        
        results.append({
            "text": chunk["text"],
            "source": chunk["source"],
            "score": round(fused[idx], 4),
        })
    
    return results


# ── CLI TEST ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    test_queries = [
        "What was the main finding of the brand study?",
        "What does the variable frm_brand_awareness measure?",
    ]
    for q in test_queries:
        print(f"\nQuery: {q}")
        print("-" * 60)
        for r in retrieve(q):
            print(f"  [{r['score']:.4f}] ({r['source']}) {r['text'][:120]}...")