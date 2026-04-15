"""
Implement your retrieval system here.

The test_files/ folder contains mixed-format documents:
  - PDFs, PowerPoint, Word, emails, JSON glossary

Your retrieve() function should handle both:
  - Broad queries: "What was the research methodology?"
  - Narrow queries: "What does 'frm_brand_awareness' measure?"
"""

import re
import json
import math
import email
from pathlib import Path
from typing import List, Dict

# ── Optional imports ──────────────────────────────────────────────────────────
try:
    import pdfplumber
    _PDF = True
except ImportError:
    _PDF = False

try:
    from pptx import Presentation
    _PPTX = True
except ImportError:
    _PPTX = False

try:
    from docx import Document as DocxDocument
    _DOCX = True
except ImportError:
    _DOCX = False

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np
    _TFIDF = True
except ImportError:
    _TFIDF = False

# sentence-transformers: optional upgrade (used if model is cached locally)
try:
    from sentence_transformers import SentenceTransformer
    import numpy as np
    _ST = True
except ImportError:
    _ST = False

TEST_FILES_DIR = Path(__file__).parent / "test_files"
CHUNK_WORDS   = 150
CHUNK_OVERLAP = 30
TOP_K         = 5

_INDEX       : List[Dict] = []
_VECTORIZER               = None
_TFIDF_MATRIX             = None
_ST_MODEL                 = None
_ST_EMBEDDINGS            = None


# ── Helpers ───────────────────────────────────────────────────────────────────

def _chunks(text: str, source: str, meta: str = "") -> List[Dict]:
    words = text.split()
    step  = CHUNK_WORDS - CHUNK_OVERLAP
    out   = []
    for i in range(0, max(1, len(words)), step):
        chunk = " ".join(words[i : i + CHUNK_WORDS]).strip()
        if len(chunk) > 30:
            out.append({"text": chunk, "source": source, "meta": meta})
    return out


def _clean(q: str) -> str:
    """Strip quotes/punctuation so 'frm_brand_awareness' matches frm_brand_awareness."""
    return re.sub(r"[^a-z0-9_ ]", "", q.lower()).strip()


def _abbreviate_var(name: str) -> List[str]:
    """
    Generate partial-name aliases for variable lookup.
    e.g. frm_brand_consideration -> [frm_consideration, frm_brand_consideration]
    Handles queries that drop the middle segment.
    """
    parts = name.split("_")
    aliases = [name]
    # Drop middle segment if 3+ parts: frm_brand_X -> frm_X
    if len(parts) >= 3:
        short = parts[0] + "_" + "_".join(parts[2:])
        aliases.append(short)
    return aliases


# ── Extractors ────────────────────────────────────────────────────────────────

def _extract_pdf(path: Path) -> List[Dict]:
    if not _PDF:
        return []
    out = []
    try:
        with pdfplumber.open(path) as pdf:
            for n, page in enumerate(pdf.pages, 1):
                t = page.extract_text() or ""
                out += _chunks(t, path.name, f"p{n}")
    except Exception:
        pass
    return out


def _extract_pptx(path: Path) -> List[Dict]:
    if not _PPTX:
        return []
    out = []
    try:
        prs = Presentation(path)
        for n, slide in enumerate(prs.slides, 1):
            parts = []
            for shape in slide.shapes:
                if shape.has_text_frame:
                    for para in shape.text_frame.paragraphs:
                        t = para.text.strip()
                        if t:
                            parts.append(t)
            out += _chunks(" ".join(parts), path.name, f"slide{n}")
    except Exception:
        pass
    return out


def _extract_docx(path: Path) -> List[Dict]:
    if not _DOCX:
        return []
    out = []
    try:
        doc   = DocxDocument(path)
        parts = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
        for table in doc.tables:
            for row in table.rows:
                rt = " | ".join(c.text.strip() for c in row.cells if c.text.strip())
                if rt:
                    parts.append(rt)
        out += _chunks(" ".join(parts), path.name, "doc")
    except Exception:
        pass
    return out


def _extract_eml(path: Path) -> List[Dict]:
    """Works on Python 3.8+ on both Windows and Linux."""
    out = []
    try:
        with open(path, "rb") as f:
            raw = f.read()
        msg    = email.message_from_bytes(raw)
        subj   = msg.get("Subject", "")
        sender = msg.get("From", "")
        date   = msg.get("Date", "")
        body   = ""
        if msg.is_multipart():
            for part in msg.walk():
                if part.get_content_type() == "text/plain":
                    payload = part.get_payload(decode=True)
                    if payload:
                        body += payload.decode("utf-8", errors="ignore")
        else:
            payload = msg.get_payload(decode=True)
            if payload:
                body = payload.decode("utf-8", errors="ignore")
        body = re.sub(r"<[^>]+>", " ", body)
        body = re.sub(r"\s+", " ", body).strip()
        full = f"Subject: {subj} | From: {sender} | Date: {date}\n{body}"
        out += _chunks(full, path.name, f"email:{subj[:40]}")
    except Exception:
        pass
    return out


def _extract_json_glossary(path: Path) -> List[Dict]:
    """
    Correctly handles glossary_partial.json structure:
      {
        "SHEET_DEFINITIONS": { sheet_name: { definition, use_for } },
        "V":                  { var_name:   { definition, type, table } }
      }

    Each entry = its own chunk for precise lookup.
    Variable entries include shortened aliases so queries like
    'frm_consideration' still match 'frm_brand_consideration'.
    """
    out = []
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        # Sheet definitions
        for sheet_name, info in data.get("SHEET_DEFINITIONS", {}).items():
            if not isinstance(info, dict):
                continue
            defn    = info.get("definition", "")
            use_for = info.get("use_for", [])
            use_str = ", ".join(use_for) if isinstance(use_for, list) else str(use_for)
            text = (
                f"Sheet: {sheet_name}. {defn} "
                f"Use for: {use_str}. "
                f"This sheet covers: {use_str}."
            )
            out.append({"text": text, "source": "glossary", "meta": f"sheet:{sheet_name}"})

        # Variable definitions with aliases
        for var_name, info in data.get("V", {}).items():
            if not isinstance(info, dict):
                continue
            defn    = info.get("definition", "")
            vtype   = info.get("type", "")
            aliases = _abbreviate_var(var_name)
            alias_str = " ".join(aliases)   # e.g. "frm_brand_consideration frm_consideration"
            # Repeat name + aliases + 'measure'/'represent' so query tokens hit
            text = (
                f"Variable: {var_name}. "
                f"The variable {alias_str} measures: {defn} "
                f"represents: {defn} "
                f"Type: {vtype}. "
                f"Definition of {alias_str}: {defn}"
            )
            out.append({"text": text, "source": "glossary", "meta": f"var:{var_name}"})

    except Exception:
        pass
    return out


# ── Index builder ─────────────────────────────────────────────────────────────

def _build_index() -> List[Dict]:
    """Scan test_files/ at runtime — no hardcoded filenames."""
    base = TEST_FILES_DIR
    if not base.exists():
        return []
    index = []
    for p in sorted(base.glob("*.pdf")):  index += _extract_pdf(Path(p))
    for p in sorted(base.glob("*.pptx")): index += _extract_pptx(Path(p))
    for p in sorted(base.glob("*.docx")): index += _extract_docx(Path(p))
    for p in sorted(base.glob("*.eml")):  index += _extract_eml(Path(p))
    for p in sorted(base.glob("*.json")): index += _extract_json_glossary(Path(p))
    return index


# ── BM25 ──────────────────────────────────────────────────────────────────────

def _tok(text: str) -> List[str]:
    return re.findall(r"[a-z0-9_]+", text.lower())


def _bm25(query: str, corpus_tokens: List[List[str]],
          k1: float = 1.5, b: float = 0.75) -> List[float]:
    n     = len(corpus_tokens)
    avgdl = sum(len(d) for d in corpus_tokens) / max(n, 1)
    qtoks = _tok(_clean(query))
    df: Dict[str, int] = {}
    for doc in corpus_tokens:
        for t in set(doc):
            df[t] = df.get(t, 0) + 1
    idf = {t: math.log((n - f + 0.5) / (f + 0.5) + 1) for t, f in df.items()}
    scores = []
    for doc in corpus_tokens:
        dl   = len(doc)
        tfm: Dict[str, int] = {}
        for t in doc:
            tfm[t] = tfm.get(t, 0) + 1
        s = sum(
            idf.get(t, 0) * tfm.get(t, 0) * (k1 + 1)
            / (tfm.get(t, 0) + k1 * (1 - b + b * dl / max(avgdl, 1)))
            for t in qtoks
        )
        scores.append(s)
    return scores


def _mm(vals: List[float]) -> List[float]:
    mn, mx = min(vals), max(vals)
    if mx == mn:
        return [0.0] * len(vals)
    return [(v - mn) / (mx - mn) for v in vals]


# ── TF-IDF cosine (sklearn, 100% local) ──────────────────────────────────────

def _tfidf(query: str, texts: List[str]) -> List[float]:
    global _VECTORIZER, _TFIDF_MATRIX
    if not _TFIDF:
        return [0.0] * len(texts)
    if _VECTORIZER is None:
        _VECTORIZER   = TfidfVectorizer(
            ngram_range=(1, 2),
            sublinear_tf=True,
            min_df=1,
            token_pattern=r"[a-z0-9_]+"
        )
        _TFIDF_MATRIX = _VECTORIZER.fit_transform(texts)
    q_vec = _VECTORIZER.transform([_clean(query)])
    return cosine_similarity(q_vec, _TFIDF_MATRIX)[0].tolist()


# ── Dense (sentence-transformers, optional upgrade) ───────────────────────────

def _dense(query: str, texts: List[str]) -> List[float]:
    global _ST_MODEL, _ST_EMBEDDINGS
    if not _ST:
        return [0.0] * len(texts)
    try:
        if _ST_MODEL is None:
            _ST_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
        if _ST_EMBEDDINGS is None:
            _ST_EMBEDDINGS = _ST_MODEL.encode(
                texts, batch_size=64, show_progress_bar=False
            )
        q_emb  = _ST_MODEL.encode([query], show_progress_bar=False)[0]
        norms  = np.linalg.norm(_ST_EMBEDDINGS, axis=1)
        q_norm = np.linalg.norm(q_emb)
        if q_norm > 0:
            sims = (_ST_EMBEDDINGS @ q_emb) / (norms * q_norm + 1e-9)
            return sims.tolist()
    except Exception:
        pass
    return [0.0] * len(texts)


# ── Query-type detection ──────────────────────────────────────────────────────

def _is_narrow(query: str) -> bool:
    q = query.lower()
    if re.search(r"[a-z]{2,}_[a-z]", q):   # variable name pattern
        return True
    narrow_kw = {
        "define", "definition", "what does", "what is the",
        "measure", "scale", "sample size", "total sample",
        "trialist", "share of wallet", "variable", "represent",
        "probable trialists", "preference rating",
    }
    return any(k in q for k in narrow_kw)


# ── Main ──────────────────────────────────────────────────────────────────────

def retrieve(query: str) -> List[Dict]:
    """
    Return top-5 most relevant passages for the query.

    Each result must have:
      - "text": str       — the passage content
      - "source": str     — source filename or "glossary"
      - "score": float    — relevance score (higher = better)
    """
    global _INDEX, _VECTORIZER, _TFIDF_MATRIX

    if not _INDEX:
        _INDEX = _build_index()
    if not _INDEX:
        return []

    texts = [c["text"] for c in _INDEX]

    # BM25 with quote-stripped query
    corpus_toks = [_tok(t) for t in texts]
    bm25_norm   = _mm(_bm25(query, corpus_toks))

    # Semantic: sentence-transformers if available, else sklearn TF-IDF
    if _ST:
        sem_norm = _mm(_dense(query, texts))
    else:
        sem_norm = _mm(_tfidf(query, texts))

    # Hybrid fusion: BM25-heavy for narrow (exact match), semantic-heavy for broad
    narrow = _is_narrow(query)
    w_bm25, w_sem = (0.65, 0.35) if narrow else (0.30, 0.70)
    fused = [w_bm25 * b + w_sem * s for b, s in zip(bm25_norm, sem_norm)]

    # Rank + deduplicate
    ranked  = sorted(range(len(fused)), key=lambda i: fused[i], reverse=True)
    results : List[Dict] = []
    seen    = set()
    for idx in ranked:
        if len(results) >= TOP_K:
            break
        key = _INDEX[idx]["text"][:80].lower().strip()
        if key in seen:
            continue
        seen.add(key)
        results.append({
            "text":   _INDEX[idx]["text"],
            "source": _INDEX[idx]["source"],
            "score":  round(fused[idx], 4),
        })
    return results