from __future__ import annotations
import os
import json
import glob
from sentence_transformers import SentenceTransformer, util
from pypdf import PdfReader
from docx import Document
from pptx import Presentation

# Small model ensures < 15 min setup and fast inference
model = SentenceTransformer('all-MiniLM-L6-v2')

def extract_text_from_file(file_path):
    """Helper to extract text from various file formats."""
    ext = file_path.lower()
    text = ""
    try:
        if ext.endswith(".pdf"):
            reader = PdfReader(file_path)
            for page in reader.pages:
                text += page.extract_text() + " "
        elif ext.endswith(".docx"):
            doc = Document(file_path)
            text = " ".join([p.text for p in doc.paragraphs])
        elif ext.endswith(".pptx"):
            prs = Presentation(file_path)
            for slide in prs.slides:
                for shape in slide.shapes:
                    if hasattr(shape, "text"):
                        text += shape.text + " "
        elif ext.endswith((".txt", ".md", ".eml")):
            with open(file_path, 'r', encoding="utf-8", errors='ignore') as f:
                text = f.read()
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
    return text

def retrieve(query: str) -> list[dict]:
    data_dir = "test_files"
    passages = []
    
    # 1. Runtime Scan
    file_paths = glob.glob(os.path.join(data_dir, "*"))
    
    for file_path in file_paths:
        fname = os.path.basename(file_path)
        
        # Handle JSON Glossary specifically (Fixed for the AttributeError)
        if file_path.endswith(".json"):
            with open(file_path, 'r', encoding="utf-8", errors='ignore') as f:
                try:
                    data = json.load(f)
                    if isinstance(data, list):
                        for item in data:
                            # FIX: Check if item is a dict before using .get()
                            if isinstance(item, dict):
                                name = item.get('name', '')
                                desc = item.get('description', '')
                                if name.lower() in query.lower():
                                    passages.append({
                                        "text": f"Variable: {name} - {desc}",
                                        "source": fname,
                                        "score": 1.0 
                                    })
                except Exception:
                    continue
        
        # Handle all other document types
        else:
            content = extract_text_from_file(file_path)
            if content.strip():
                # For Task 2, we chunk the text to keep it relevant
                chunk = content[:1500] 
                passages.append({
                    "text": chunk,
                    "source": fname,
                    "score": 0.5 # Default score for now
                })

    # 2. Simple Re-ranking using the embedding model
    if passages:
        texts = [p["text"] for p in passages]
        corpus_embeddings = model.encode(texts, convert_to_tensor=True)
        query_embedding = model.encode(query, convert_to_tensor=True)
        hits = util.semantic_search(query_embedding, corpus_embeddings, top_k=5)[0]
        
        results = []
        for hit in hits:
            idx = hit['corpus_id']
            res = passages[idx].copy()
            res['score'] = float(hit['score'])
            results.append(res)
        return results
    
    return []

if __name__ == "__main__":
    results = retrieve("What does frm_brand_awareness measure?")
    print(json.dumps(results, indent=2))