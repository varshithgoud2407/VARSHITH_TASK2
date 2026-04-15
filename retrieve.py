"""
Hybrid retrieval system with:
- Recursive glossary extraction from nested JSON
- Glossary prioritisation for narrow variable queries
- Email boosting for sender-style queries
- BM25 + dense retrieval with RRF fusion
"""

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import pdfplumber
except ImportError:
    pdfplumber = None

try:
    from pptx import Presentation
except ImportError:
    Presentation = None

try:
    from docx import Document
except ImportError:
    Document = None

try:
    from email import policy
    from email.parser import BytesParser
except ImportError:
    policy = None
    BytesParser = None

try:
    from rank_bm25 import BM25Okapi
except ImportError:
    BM25Okapi = None

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None


logger = logging.getLogger(__name__)

MIN_CHUNK_CHARS = 50
PREFIXES = ("frm_", "var_", "q_")

_CORPUS: List[Dict[str, Any]] = []
_BM25: Any = None
_EMBEDDINGS: Optional[np.ndarray] = None
_EMBEDDER: Any = None
_GLOSSARY: Dict[str, str] = {}


def _extract_from_pdf(path: Path) -> Tuple[str, Dict[str, str]]:
    if pdfplumber is None:
        return "", {}
    text: List[str] = []
    try:
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text.append(page_text)
    except Exception as exc:
        logger.warning("Error parsing PDF %s: %s", path, exc)
    return "\n".join(text), {}


def _extract_from_pptx(path: Path) -> Tuple[str, Dict[str, str]]:
    if Presentation is None:
        return "", {}
    text: List[str] = []
    try:
        prs = Presentation(path)
        for slide in prs.slides:
            for shape in slide.shapes:
                if hasattr(shape, "text") and shape.text:
                    text.append(shape.text)
    except Exception as exc:
        logger.warning("Error parsing PPTX %s: %s", path, exc)
    return "\n".join(text), {}


def _extract_from_docx(path: Path) -> Tuple[str, Dict[str, str]]:
    if Document is None:
        return "", {}
    text: List[str] = []
    try:
        doc = Document(path)
        for para in doc.paragraphs:
            if para.text:
                text.append(para.text)
    except Exception as exc:
        logger.warning("Error parsing DOCX %s: %s", path, exc)
    return "\n".join(text), {}


def _extract_from_eml(path: Path) -> Tuple[str, Dict[str, str]]:
    if BytesParser is None or policy is None:
        return "", {}
    text: List[str] = []
    try:
        with open(path, "rb") as f:
            msg = BytesParser(policy=policy.default).parse(f)
        subject = msg.get("Subject", "")
        if subject:
            text.append(f"Subject: {subject}")
        body = None
        if msg.is_multipart():
            for part in msg.walk():
                if part.get_content_type() == "text/plain":
                    body = part.get_content()
                    break
        else:
            body = msg.get_content()
        if body:
            text.append(body)
    except Exception as exc:
        logger.warning("Error parsing EML %s: %s", path, exc)
    return "\n".join(text), {}


def _normalize_glossary_key(value: str) -> str:
    normalized = value.strip().lower().replace("-", " ").replace("/", " ")
    normalized = normalized.replace("_", " ")
    return re.sub(r"\s+", " ", normalized)


def _is_placeholder_definition(key: str, definition: str) -> bool:
    return _normalize_glossary_key(key) == _normalize_glossary_key(definition)


def _store_glossary_entry(
    glossary: Dict[str, str], full_lines: List[str], key: str, definition: str
) -> None:
    cleaned_key = key.strip().lower()
    cleaned_definition = definition.strip()
    if not cleaned_key or not cleaned_definition:
        return
    glossary[cleaned_key] = cleaned_definition
    full_lines.append(f"{key}: {cleaned_definition}")


def _extract_glossary_entries(
    data: Any, glossary: Dict[str, str], full_lines: List[str]
) -> None:
    if not isinstance(data, dict):
        return

    for key, value in data.items():
        if not isinstance(value, dict):
            continue
        definition = value.get("definition")
        if isinstance(definition, str):
            _store_glossary_entry(glossary, full_lines, key, definition)
        _extract_glossary_entries(value, glossary, full_lines)


def _extract_from_json_glossary(path: Path) -> Tuple[str, Dict[str, str]]:
    glossary: Dict[str, str] = {}
    full_lines: List[str] = []
    data = None

    for encoding in ("utf-8", "latin-1"):
        try:
            with open(path, "r", encoding=encoding) as f:
                data = json.load(f)
            break
        except (UnicodeDecodeError, json.JSONDecodeError):
            continue
        except Exception as exc:
            logger.warning("Error loading glossary %s: %s", path, exc)
            return "", glossary

    if data is None:
        logger.warning("Could not decode glossary file %s", path)
        return "", glossary

    _extract_glossary_entries(data, glossary, full_lines)
    return "\n".join(full_lines), glossary


def _extract_text_from_file(file_path: Path) -> Tuple[str, Dict[str, str]]:
    suffix = file_path.suffix.lower()
    if suffix == ".pdf":
        return _extract_from_pdf(file_path)
    if suffix == ".pptx":
        return _extract_from_pptx(file_path)
    if suffix == ".docx":
        return _extract_from_docx(file_path)
    if suffix == ".eml":
        return _extract_from_eml(file_path)
    if suffix == ".json" and "glossary" in file_path.name.lower():
        return _extract_from_json_glossary(file_path)
    return "", {}


def _chunk_text(text: str, chunk_size: int = 300, overlap: int = 50) -> List[str]:
    if not text:
        return []
    words = text.split()
    if len(words) <= chunk_size:
        return [text] if len(text.strip()) >= MIN_CHUNK_CHARS else []

    chunks: List[str] = []
    step = chunk_size - overlap
    for start in range(0, len(words), step):
        chunk = " ".join(words[start : start + chunk_size])
        if len(chunk.strip()) >= MIN_CHUNK_CHARS:
            chunks.append(chunk)
        if start + chunk_size >= len(words):
            break
    return chunks


def _build_corpus(data_dir: str = "test_files") -> None:
    global _CORPUS, _GLOSSARY, _BM25, _EMBEDDINGS, _EMBEDDER
    if _CORPUS:
        return

    data_path = Path(data_dir)
    if not data_path.exists():
        module_data_path = Path(__file__).resolve().parent / "test_files"
        if module_data_path.exists():
            data_path = module_data_path
        elif (Path(".") / "test_files").exists():
            data_path = Path(".") / "test_files"
        else:
            data_path = Path(__file__).resolve().parent

    all_chunks: List[Dict[str, Any]] = []
    glossary_global: Dict[str, str] = {}
    chunk_id = 0

    for file_path in data_path.glob("*"):
        if file_path.is_dir():
            continue
        text, file_glossary = _extract_text_from_file(file_path)
        if file_glossary:
            glossary_global.update(file_glossary)
        if not text:
            continue
        for chunk in _chunk_text(text):
            all_chunks.append(
                {
                    "id": chunk_id,
                    "text": chunk,
                    "source": file_path.name,
                }
            )
            chunk_id += 1

    if not all_chunks:
        raise RuntimeError(f"No documents loaded from {data_path}")

    _CORPUS = all_chunks
    _GLOSSARY = glossary_global

    logger.info("Loaded glossary with %s entries", len(_GLOSSARY))
    if not _GLOSSARY:
        logger.warning("Glossary is empty; narrow variable lookup will be degraded")

    if BM25Okapi is not None:
        tokenized = [doc["text"].lower().split() for doc in _CORPUS]
        _BM25 = BM25Okapi(tokenized)

    if SentenceTransformer is not None:
        try:
            _EMBEDDER = SentenceTransformer("all-MiniLM-L6-v2")
            texts = [doc["text"] for doc in _CORPUS]
            _EMBEDDINGS = _EMBEDDER.encode(texts, show_progress_bar=False)
        except Exception as exc:
            logger.warning("Dense embedder unavailable, continuing with BM25 only: %s", exc)
            _EMBEDDER = None
            _EMBEDDINGS = None


def _dense_retrieve(query: str, top_k: int = 20) -> List[Tuple[int, float]]:
    if _EMBEDDINGS is None or _EMBEDDER is None or not _CORPUS:
        return []
    qvec = _EMBEDDER.encode([query])[0]
    norms = np.linalg.norm(_EMBEDDINGS, axis=1)
    qnorm = np.linalg.norm(qvec)
    if qnorm == 0:
        return []
    sim = np.dot(_EMBEDDINGS, qvec) / (norms * qnorm + 1e-8)
    top = np.argsort(sim)[-top_k:][::-1]
    return [(int(idx), float(sim[idx])) for idx in top]


def _bm25_retrieve(query: str, top_k: int = 20) -> List[Tuple[int, float]]:
    if _BM25 is None or not _CORPUS:
        return []
    scores = _BM25.get_scores(query.lower().split())
    top = np.argsort(scores)[-top_k:][::-1]
    return [(int(idx), float(scores[idx])) for idx in top]


def _apply_boosts(
    query: str,
    bm25_results: List[Tuple[int, float]],
    dense_results: List[Tuple[int, float]],
) -> Tuple[List[Tuple[int, float]], List[Tuple[int, float]]]:
    query_lower = query.lower()
    is_email_query = any(
        keyword in query_lower
        for keyword in ["email", "sent", "sender", "who sent", "from:", "subject:"]
    )
    is_broad_query = not _is_narrow_query(query)

    def boost_list(results: List[Tuple[int, float]]) -> List[Tuple[int, float]]:
        boosted: List[Tuple[int, float]] = []
        for doc_id, score in results:
            source = _CORPUS[doc_id]["source"]
            if is_email_query and source.endswith(".eml"):
                score *= 1.5
            if is_broad_query and source == "glossary_partial.json":
                score *= 0.5
            boosted.append((doc_id, score))
        boosted.sort(key=lambda item: item[1], reverse=True)
        return boosted

    return boost_list(bm25_results), boost_list(dense_results)


def _is_narrow_query(query: str) -> bool:
    query_lower = query.lower()
    if re.search(r"\b(?:frm|var|q)_[a-z0-9_]{3,}\b", query_lower):
        return True
    if re.search(r"\b[a-z]+_[a-z0-9_]{4,}\b", query_lower):
        return True
    if re.search(r"(?:what does|define|meaning of|measure|scale|represent)\s+[`'\"]?[^?.!,]+", query_lower):
        return True
    return bool(re.search(r"[`'\"][a-z0-9_ ][^`'\"]*[`'\"]", query_lower))


def _rrf_fusion(lists: List[List[Tuple[int, float]]], k: int = 60) -> List[Tuple[int, float]]:
    fused: Dict[int, float] = {}
    for results in lists:
        for rank, (doc_id, _) in enumerate(results):
            fused[doc_id] = fused.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
    return sorted(fused.items(), key=lambda item: item[1], reverse=True)


def _candidate_terms_from_query(query: str) -> List[str]:
    query_lower = query.lower()
    candidates: List[str] = []

    for match in re.findall(r"[`'\"]([^`'\"]+)[`'\"]", query_lower):
        cleaned = match.strip()
        if cleaned:
            candidates.append(cleaned)

    for match in re.findall(r"\b[a-z][a-z0-9_]{3,}\b", query_lower):
        if "_" in match:
            candidates.append(match)

    phrase_match = re.search(
        r"(?:what does|define|meaning of|measure|scale|represent)\s+[`'\"]?([^?.!,]+)",
        query_lower,
    )
    if phrase_match:
        cleaned = phrase_match.group(1).strip(" `\"'")
        if cleaned:
            candidates.append(cleaned)

    seen = set()
    unique_candidates: List[str] = []
    for candidate in candidates:
        normalized = re.sub(r"\s+", " ", candidate.strip())
        if normalized and normalized not in seen:
            unique_candidates.append(normalized)
            seen.add(normalized)
    return unique_candidates


def _try_glossary_lookup(query: str) -> Optional[List[Dict[str, Any]]]:
    if not _GLOSSARY or not _is_narrow_query(query):
        return None

    candidates = _candidate_terms_from_query(query)
    if not candidates:
        return None

    best_match = None
    best_score = -1

    for candidate in candidates:
        raw_candidate = candidate.lower()
        normalized_candidate = _normalize_glossary_key(candidate)
        stripped_variants = [raw_candidate, normalized_candidate]

        for prefix in PREFIXES:
            if raw_candidate.startswith(prefix):
                stripped = raw_candidate[len(prefix) :]
                stripped_variants.extend([stripped, _normalize_glossary_key(stripped)])

        for key, description in _GLOSSARY.items():
            key_normalized = _normalize_glossary_key(key)
            score = -1

            if raw_candidate == key or normalized_candidate == key_normalized:
                score = 300
            elif any(variant and (key == variant or key_normalized == variant) for variant in stripped_variants):
                score = 275
            elif any(
                variant and (key.endswith(f"_{variant}") or key_normalized.endswith(f" {variant}"))
                for variant in stripped_variants
            ):
                score = 245
            elif any(variant and variant in key_normalized for variant in stripped_variants):
                score = 215
            elif normalized_candidate in key_normalized or key_normalized in normalized_candidate:
                score = 200

            if score < 0:
                continue

            if _is_placeholder_definition(key, description):
                score -= 120
            else:
                score += 15
            score += min(len(description.strip()) // 40, 10)

            if score > best_score:
                best_score = score
                best_match = (key, description, candidate, score)

    if best_match is None:
        return None

    key, description, candidate, score = best_match
    exactish = score >= 300 or _normalize_glossary_key(key) == _normalize_glossary_key(candidate)
    confidence = 1.0 if exactish else 0.95
    if exactish:
        text = f"Variable '{key}' measures: {description}"
    else:
        text = f"Variable '{key}' (close match to '{candidate}'): {description}"
    return [{"text": text, "source": "glossary", "score": confidence}]


def retrieve(query: str) -> List[Dict[str, Any]]:
    _build_corpus()

    glossary_result = _try_glossary_lookup(query)
    if glossary_result:
        bm25_results = _bm25_retrieve(query, top_k=20)
        dense_results = _dense_retrieve(query, top_k=20)
        if bm25_results or dense_results:
            bm25_boosted, dense_boosted = _apply_boosts(query, bm25_results, dense_results)
            fused = _rrf_fusion([bm25_boosted, dense_boosted])
            if fused:
                max_score = fused[0][1]
                min_score = fused[-1][1] if len(fused) > 1 else max_score
                if max_score == min_score:
                    norm_scores = [1.0] * len(fused)
                else:
                    norm_scores = [(score - min_score) / (max_score - min_score) for _, score in fused]

                remaining: List[Dict[str, Any]] = []
                for index, (doc_id, _) in enumerate(fused[:4]):
                    chunk = _CORPUS[doc_id]
                    remaining.append(
                        {
                            "text": chunk["text"],
                            "source": chunk["source"],
                            "score": norm_scores[index],
                        }
                    )
                return glossary_result + remaining
        return glossary_result

    bm25_results = _bm25_retrieve(query, top_k=20)
    dense_results = _dense_retrieve(query, top_k=20)

    if not bm25_results and not dense_results:
        query_terms = set(query.lower().split())
        fallback: List[Tuple[int, float]] = []
        for idx, chunk in enumerate(_CORPUS):
            text_lower = chunk["text"].lower()
            score = sum(1 for term in query_terms if term in text_lower) / (len(query_terms) + 1e-6)
            if score > 0:
                fallback.append((idx, score))
        fallback.sort(key=lambda item: item[1], reverse=True)
        return [
            {
                "text": _CORPUS[idx]["text"],
                "source": _CORPUS[idx]["source"],
                "score": score,
            }
            for idx, score in fallback[:5]
        ]

    bm25_boosted, dense_boosted = _apply_boosts(query, bm25_results, dense_results)
    fused = _rrf_fusion([bm25_boosted, dense_boosted])
    if not fused:
        return []

    max_score = fused[0][1]
    min_score = fused[-1][1] if len(fused) > 1 else max_score
    if max_score == min_score:
        norm_scores = [1.0] * len(fused)
    else:
        norm_scores = [(score - min_score) / (max_score - min_score) for _, score in fused]

    return [
        {
            "text": _CORPUS[doc_id]["text"],
            "source": _CORPUS[doc_id]["source"],
            "score": norm_scores[index],
        }
        for index, (doc_id, _) in enumerate(fused[:5])
    ]
