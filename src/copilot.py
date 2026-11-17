"""
MacroOps AI - AI Operations Copilot.

Tool-calling agent (LangChain + Ollama qwen2.5:3b) with memory.

  Short-term memory : last turns of this chat
  Long-term memory  : facts kept in data/processed/copilot_memory.json
                      (your store, your role, notes). Extracted by rules, not by the
                      LLM, so a small model cannot corrupt it.

Guard rails for a small model
  - tools only: numbers must come from tool results
  - nudge: if a data question gets no tool call, ask once more to use a tool
  - max 4 tool rounds per question, tool errors are returned to the model as text
  - tool arguments are normalised (st95 -> ST095)

CLI:  python -m src.copilot
"""
import json
import re
import time

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from src.config import PROCESSED
from src.tools import TOOLS, run_tool

LLM_MODEL = "qwen2.5:3b"
MAX_ROUNDS = 4
HISTORY_TURNS = 6
MEMORY_PATH = PROCESSED / "copilot_memory.json"

SYSTEM = """You are MacroOps Copilot, an operations assistant for quick-commerce dark stores.
RULES:
1. For ANY number, KPI, store, order or risk question, call a tool. Never guess numbers.
2. For procedures, policies, thresholds, 'what should we do' questions, call search_sop and answer from its excerpts. Cite sources like [SOP-004].
3. Use only tool results. If a tool returns an error or nothing, say so plainly.
4. Reply in 2 to 5 short sentences or a short list. Quote the exact numbers from the tool.
5. Data covers 1-30 Sep 2026 and is synthetic.
6. If the question is not about operations, say you only help with MacroOps operations.
{memory}"""

DATA_WORDS = re.compile(
    r"kpi|otd|on-time|on time|breach|late|delay|store|order|ord-|risk|rate|worst|best|compare|"
    r"how many|average|stockout|cancel|picker|rider|headcount|peak|sla|what should|procedure|policy|"
    r"escalat|compensat|incident|rain", re.I)


def get_chat_llm():
    from langchain_ollama import ChatOllama
    return ChatOllama(model=LLM_MODEL, temperature=0, num_ctx=8192)


# ------------------------------------------------------------------ long-term memory
class Memory:
    ROLES = ["dispatch lead", "dispatcher", "store manager", "area manager", "supervisor",
             "inventory lead", "rider coordinator", "operations head"]

    def __init__(self, path=MEMORY_PATH):
        self.path = path
        self.facts = {"store_id": "", "role": "", "notes": []}
        if path.exists():
            try:
                self.facts.update(json.loads(path.read_text()))
            except Exception:
                pass

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.facts, indent=2))

    def update_from(self, text: str) -> list[str]:
        """Rule-based fact extraction. Returns what changed."""
        changed, t = [], text.strip()
        if re.search(r"\bforget (everything|all|my)", t, re.I):
            self.facts = {"store_id": "", "role": "", "notes": []}
            self.save()
            return ["memory cleared"]
        m = re.search(r"\b(?:my store is|i manage|i run|i work (?:at|in)|i(?:'m| am) (?:at|in))\s+(?:store\s+)?(st\s?-?\d{1,3})\b", t, re.I)
        if m:
            sid = "ST%03d" % int(re.search(r"\d+", m.group(1)).group())
            if sid != self.facts["store_id"]:
                self.facts["store_id"] = sid
                changed.append(f"store = {sid}")
        m = re.search(r"\bi(?:'m| am)\s+(?:the |a |an )?(%s)\b" % "|".join(self.ROLES), t, re.I)
        if m and m.group(1).lower() != self.facts["role"]:
            self.facts["role"] = m.group(1).lower()
            changed.append(f"role = {self.facts['role']}")
        m = re.search(r"\bremember (?:that )?(.+)", t, re.I)
        if m:
            note = m.group(1).strip().rstrip(".")[:200]
            if note and note not in self.facts["notes"]:
                self.facts["notes"] = (self.facts["notes"] + [note])[-5:]
                changed.append(f"note = {note}")
        if changed:
            self.save()
        return changed

    def prompt_text(self) -> str:
        f, lines = self.facts, []
        if f["store_id"]:
            lines.append(f"The user manages store {f['store_id']}. When they say 'my store', use store_id={f['store_id']}.")
        if f["role"]:
            lines.append(f"The user's role is {f['role']}. Tailor advice to that role.")
        for n in f["notes"]:
            lines.append(f"User note: {n}")
        return ("\nMEMORY:\n" + "\n".join(lines)) if lines else ""


# ------------------------------------------------------------------ copilot
class Copilot:
    def __init__(self, llm=None, memory: Memory | None = None):
        self.llm = llm or get_chat_llm()
        self.llm_tools = self.llm.bind_tools(TOOLS)
        self.memory = memory or Memory()
        self.history: list = []          # Human/AI messages only (tool chatter dropped)

    def reset_chat(self):
        self.history = []

    def chat(self, text: str) -> dict:
        t0 = time.time()
        changed = self.memory.update_from(text)
        system = SystemMessage(SYSTEM.format(memory=self.memory.prompt_text()))
        messages = [system, *self.history[-2 * HISTORY_TURNS:], HumanMessage(text)]
        used, nudged = [], False

        for _ in range(MAX_ROUNDS):
            ai = self.llm_tools.invoke(messages)
            messages.append(ai)
            calls = getattr(ai, "tool_calls", None) or []
            if not calls:
                if not used and not nudged and DATA_WORDS.search(text) and (not changed or "?" in text):
                    nudged = True
                    messages.append(HumanMessage(
                        "Use a tool to get this information. Do not answer from memory."))
                    continue
                break
            for i, tc in enumerate(calls):
                result = run_tool(tc["name"], tc.get("args") or {})
                used.append({"name": tc["name"], "args": tc.get("args") or {}, "result": result})
                messages.append(ToolMessage(content=result, tool_call_id=tc.get("id") or f"call_{i}",
                                            name=tc["name"]))
        else:
            ai = self.llm.invoke(messages + [HumanMessage(
                "Now answer the question in a few sentences using only the tool results above.")])

        answer = (getattr(ai, "content", "") or "").strip()
        if not answer and used:
            answer = "I got the data but could not summarise it. Raw result: " + used[-1]["result"][:400]
        if not answer:
            answer = "I could not produce an answer. Please rephrase."
        if changed:
            answer = f"Noted ({'; '.join(changed)}). " + answer

        self.history += [HumanMessage(text), AIMessage(answer)]
        return {"answer": answer, "tools": used, "memory_changes": changed,
                "seconds": round(time.time() - t0, 1)}


# ------------------------------------------------------------------ CLI
def main():
    cp = Copilot()
    print("MacroOps Copilot. Ask about KPIs, stores, orders or SOPs. 'exit' to quit.")
    if cp.memory.prompt_text():
        print(cp.memory.prompt_text().strip())
    while True:
        try:
            q = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if q.lower() in {"exit", "quit", ""}:
            break
        r = cp.chat(q)
        for t in r["tools"]:
            print(f"  [tool] {t['name']}({t['args']})")
        print("\n" + r["answer"] + f"\n  ({r['seconds']}s)")


if __name__ == "__main__":
    main()