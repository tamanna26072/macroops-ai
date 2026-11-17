"""
MacroOps AI - RAG evaluation (UAT set).

    python -m src.rag_eval            # retrieval only (fast, no LLM)
    python -m src.rag_eval --llm      # also generate answers and check expected facts

Metrics
  hit@k       expected SOP is in the top-k retrieved sources
  MRR         1 / rank of the expected SOP (0 if missing)
  refusal     out-of-scope question correctly returns no sources
  fact match  (--llm) expected fact appears in the generated answer
"""
import argparse

import pandas as pd

from src.config import OUTPUTS
from src.rag import TOP_K, ask, get_embeddings, get_llm, get_store, retrieve

# (question, expected SOP, expected fact in answer). SOP = None means out of scope.
CASES = [
    ("What is the on-time delivery target?", "SOP-001", "90"),
    ("When does a store get escalated to the area manager?", "SOP-001", "80"),
    ("How many extra pickers should be added on promotion days?", "SOP-002", "20"),
    ("When is a store considered understaffed?", "SOP-002", "0.18"),
    ("What should a picker do when an item is missing?", "SOP-003", "60 seconds"),
    ("When do we have to call the customer about missing items?", "SOP-003", "30 percent"),
    ("What should the dispatcher do if rider assignment takes 12 minutes?", "SOP-004", "nearest"),
    ("How do we handle riders with a high breach rate?", "SOP-004", "40"),
    ("What compensation does a customer get if delivery is 15 minutes late?", "SOP-005", "10 percent"),
    ("What is the procedure for a failed delivery?", "SOP-005", "5 minutes"),
    ("How long should picking take per item?", "SOP-006", "1 minute"),
    ("What should a supervisor do when the AI flags an order as high risk after picking?", "SOP-006", "dispatcher"),
    ("What does item fill rate measure?", "SOP-007", "picked"),
    ("How is inventory accuracy calculated?", "SOP-007", "physical"),
    ("What does a HIGH risk alert mean?", "SOP-008", "threshold"),
    ("What should we do when more than 30 percent of a store's orders are flagged?", "SOP-008", "area manager"),
    ("What should be done when an order is flagged right after it is placed?", "SOP-008", "first"),
    ("What to do during heavy rain?", "SOP-009", "5 minutes"),
    ("What should we do if the order system goes down?", "SOP-009", "manual"),
    ("Who handles stockouts and replenishment problems?", "SOP-010", "inventory lead"),
    ("How fast must the area manager respond to an escalation?", "SOP-010", "30 minutes"),
    ("How accurate is the SLA risk model?", "SOP-011", "0.92"),
    ("What are the limitations of the SLA risk model?", "SOP-011", "synthetic"),
    ("What is the capital of France?", None, None),
    ("Write a poem about the ocean", None, None),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--llm", action="store_true")
    ap.add_argument("--k", type=int, default=TOP_K)
    args = ap.parse_args()

    store = get_store(get_embeddings())
    llm = get_llm() if args.llm else None
    rows = []
    for q, sop, fact in CASES:
        hits = retrieve(q, args.k, store)
        ids = [d.metadata["doc_id"] for d, _ in hits]
        uniq = list(dict.fromkeys(ids))
        row = {"question": q, "expected": sop or "none", "retrieved": ",".join(uniq),
               "top_distance": round(hits[0][1], 3) if hits else None}
        if sop:
            row["hit"] = int(sop in uniq)
            row["rr"] = 1 / (uniq.index(sop) + 1) if sop in uniq else 0.0
        else:
            row["refused"] = int(len(hits) == 0)
        if llm:
            ans = ask(q, args.k, store, llm)["answer"]
            row["answer"] = ans
            if fact:
                row["fact_ok"] = int(fact.lower() in ans.lower())
            else:
                row["fact_ok"] = int("could not find" in ans.lower())
        rows.append(row)

    df = pd.DataFrame(rows)
    out = OUTPUTS / "rag_eval.csv"
    df.to_csv(out, index=False)

    inscope = df[df.expected != "none"]
    oos = df[df.expected == "none"]
    print(f"in-scope questions: {len(inscope)} | hit@{args.k}: {inscope.hit.mean():.0%} | MRR: {inscope.rr.mean():.2f}")
    print(f"out-of-scope questions: {len(oos)} | correctly refused: {oos.refused.mean():.0%}")
    if llm:
        print(f"answer fact match: {df.fact_ok.mean():.0%} ({int(df.fact_ok.sum())}/{len(df)})")
    miss = inscope[inscope.hit == 0]
    if len(miss):
        print("\nretrieval misses:")
        print(miss[["question", "expected", "retrieved"]].to_string(index=False))
    print("\nsaved:", out)


if __name__ == "__main__":
    main()