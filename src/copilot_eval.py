"""
MacroOps AI - Copilot evaluation (tool choice + answer facts).

    python -X utf8 -m src.copilot_eval

Metrics
  tool accuracy   the expected tool was called (None = no tool expected)
  fact match      expected number / word appears in the answer
  avg seconds     latency per question
Multi-turn cases test memory ("my store is ST095" then "my store").
"""
import tempfile
from pathlib import Path

import pandas as pd

from src.config import OUTPUTS
from src.copilot import Copilot, Memory

# (turns, expected tool on LAST turn, acceptable facts in last answer)
CASES = [
    (["What is the on-time delivery rate?"], "get_kpi_summary", ("72.8", "72.9")),
    (["What is the SLA breach rate during peak hours?"], "get_kpi_summary", ("39.4",)),
    (["Which stores have the highest breach rate?"], "worst_stores", ("ST095",)),
    (["Give me a report on store ST066"], "store_report", ("58.7", "58.8", "ST066")),
    (["How is store ST095 doing?"], "store_report", ("67.6", "67.7", "ST095")),
    (["Show me the 3 riskiest orders"], "risky_orders", ("ORD-",)),
    (["Why is order ORD-082529 at risk?"], "explain_order_risk", ("ORD-082529", "risk")),
    (["What should the dispatcher do if rider assignment takes 12 minutes?"], "search_sop", ("nearest",)),
    (["What compensation does a customer get when delivery is 15 minutes late?"], "search_sop", ("10",)),
    (["What to do during heavy rain?"], "search_sop", ("5 minutes", "5 min")),
    (["Who handles inventory problems?"], "search_sop", ("inventory lead",)),
    (["My store is ST095", "What is my store's breach rate?"], "store_report", ("67.6", "67.7")),
    (["What is the capital of France?"], None, ("only help", "operations", "cannot", "can't")),
]


def main():
    rows = []
    for turns, exp_tool, facts in CASES:
        mem = Memory(Path(tempfile.mkdtemp()) / "mem.json")      # fresh memory per case
        cp = Copilot(memory=mem)
        for i, q in enumerate(turns):
            r = cp.chat(q)
        names = [t["name"] for t in r["tools"]]
        tool_ok = (exp_tool in names) if exp_tool else (len(names) == 0)
        fact_ok = any(f.lower() in r["answer"].lower() for f in facts)
        rows.append({"question": turns[-1], "expected_tool": exp_tool or "none", "tools_used": ",".join(names),
                     "tool_ok": int(tool_ok), "fact_ok": int(fact_ok), "seconds": r["seconds"],
                     "answer": r["answer"]})
        print(f"{'OK ' if tool_ok else 'BAD'} tool | {'OK ' if fact_ok else 'BAD'} fact | {r['seconds']:>5}s | {turns[-1][:60]}")

    df = pd.DataFrame(rows)
    out = OUTPUTS / "copilot_eval.csv"
    df.to_csv(out, index=False)
    print(f"\ntool accuracy: {df.tool_ok.mean():.0%} ({df.tool_ok.sum()}/{len(df)})"
          f" | fact match: {df.fact_ok.mean():.0%} | avg latency: {df.seconds.mean():.1f}s")
    print("saved:", out)


if __name__ == "__main__":
    main()