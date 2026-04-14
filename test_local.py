"""
test_local.py — Local evaluation harness for retrieve.py
Runs 10 sample queries with a basic keyword overlap scorer.
Do not over-optimise for this scorer — it's a sanity check only.
"""

from retrieve import retrieve

SAMPLE_QUERIES = [
    "What was the main finding of the brand study?",
    "What does the variable frm_brand_awareness measure?",
    "What are the key purchase barriers for frozen meals?",
    "Who sent the email about the survey questionnaire?",
    "What is the sample size of the Stocked survey?",
    "Which brands have the highest aided awareness?",
    "What are the growth levers identified in Phase 2?",
    "Explain the methodology used in the research proposal",
    "What does the frm_consideration variable represent?",
    "What channels offer the most potential for market expansion?",
]


def keyword_score(result_text: str, query: str) -> float:
    """Basic keyword overlap scorer — sanity check only."""
    query_words = set(query.lower().split())
    result_words = set(result_text.lower().split())
    stop = {"the", "a", "an", "is", "are", "was", "what", "does", "for", "of",
            "in", "and", "to", "with", "that", "this", "it", "be", "have"}
    query_keywords = query_words - stop
    if not query_keywords:
        return 0.0
    overlap = len(query_keywords & result_words)
    return round(overlap / len(query_keywords), 3)


def run_eval():
    print("=" * 70)
    print("KEA RETRIEVAL SYSTEM — LOCAL EVALUATION")
    print("=" * 70)
    
    total_score = 0.0
    
    for i, query in enumerate(SAMPLE_QUERIES, 1):
        print(f"\nQ{i:02d}: {query}")
        results = retrieve(query)
        
        if not results:
            print("  ⚠ No results returned")
            continue
        
        q_score = 0.0
        for j, r in enumerate(results, 1):
            ks = keyword_score(r["text"], query)
            q_score = max(q_score, ks)
            print(f"  [{j}] score={r['score']:.4f} | kw={ks:.3f} | "
                  f"src={r['source']} | {r['text'][:80]}...")
        
        total_score += q_score
        print(f"  → Best keyword overlap: {q_score:.3f}")
    
    avg = total_score / len(SAMPLE_QUERIES)
    print("\n" + "=" * 70)
    print(f"AVERAGE KEYWORD SCORE: {avg:.3f}  (sanity check — not ground truth)")
    print("=" * 70)


if __name__ == "__main__":
    run_eval()