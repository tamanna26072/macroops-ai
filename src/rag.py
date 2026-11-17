"""
MacroOps AI - RAG over operations SOPs.

    LangChain + ChromaDB + HuggingFace embeddings + Ollama LLM

Run from repo root:
    python -m src.rag --build                    # (re)build the vector index
    python -m src.rag "What to do if rider assignment takes 12 minutes?"
    python -m src.rag                            # interactive chat

Reused later by the Copilot tool `search_sop`.
"""
import argparse
import re
import sys

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import ROOT

KB_DIR = ROOT / "knowledge_base"
CHROMA_DIR = ROOT / "chroma_db"
COLLECTION = "macroops_sops"

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "qwen2.5:3b"

TOP_K = 4
MAX_DISTANCE = 0.75        # cosine distance above this = not relevant
MAX_SECTION_CHARS = 900

PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are the MacroOps AI operations assistant for quick-commerce dark stores.\n"
     "Answer ONLY from the SOP excerpts below. If the excerpts do not contain the answer, "
     "reply exactly: \"I could not find this in the SOPs.\"\n"
     "Be short and practical. Use numbered steps for procedures. Keep numbers and thresholds exactly as written.\n"
     "After each fact, cite its source in square brackets like [SOP-004 Rider Assignment].\n\n"
     "SOP excerpts:\n{context}"),
    ("human", "{question}"),
])


# ------------------------------------------------------------------ models (lazy imports)
def get_embeddings():
    from langchain_huggingface import HuggingFaceEmbeddings
    return HuggingFaceEmbeddings(model_name=EMBED_MODEL,
                                 encode_kwargs={"normalize_embeddings": True})


def get_llm():
    from langchain_ollama import ChatOllama
    return ChatOllama(model=LLM_MODEL, temperature=0.1, num_ctx=4096)


# ------------------------------------------------------------------ load + chunk
def _doc_meta(text: str, fname: str):
    title = re.search(r"^# (.+)$", text, re.M)
    doc_id = re.search(r"Document ID:\s*(SOP-\d+)", text)
    return (title.group(1).strip() if title else fname,
            doc_id.group(1) if doc_id else fname)


def load_chunks() -> list[Document]:
    """One chunk per '## section'. Long sections are split further."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=MAX_SECTION_CHARS, chunk_overlap=100)
    chunks = []
    for path in sorted(KB_DIR.glob("*.md")):
        text = path.read_text(encoding="utf-8")
        title, doc_id = _doc_meta(text, path.name)
        for block in re.split(r"(?m)^## ", text)[1:]:
            heading, _, body = block.partition("\n")
            heading, body = heading.strip(), body.strip()
            if not body:
                continue
            content = f"{title} - {heading}\n{body}"
            meta = {"source": path.name, "doc_id": doc_id, "title": title, "section": heading}
            if len(content) <= MAX_SECTION_CHARS:
                chunks.append(Document(page_content=content, metadata=meta))
            else:
                for part in splitter.split_text(content):
                    chunks.append(Document(page_content=part, metadata=meta))
    return chunks


# ------------------------------------------------------------------ vector store
def get_store(embeddings=None) -> Chroma:
    return Chroma(collection_name=COLLECTION, persist_directory=str(CHROMA_DIR),
                  embedding_function=embeddings or get_embeddings(),
                  collection_metadata={"hnsw:space": "cosine"})


def build_index(embeddings=None, rebuild: bool = True) -> Chroma:
    store = get_store(embeddings)
    if rebuild:
        store.delete_collection()
        store = get_store(embeddings)
    chunks = load_chunks()
    store.add_documents(chunks, ids=[f"{c.metadata['doc_id']}-{i}" for i, c in enumerate(chunks)])
    print(f"indexed {len(chunks)} chunks from {len(list(KB_DIR.glob('*.md')))} documents")
    return store


# ------------------------------------------------------------------ retrieve + answer
def retrieve(question: str, k: int = TOP_K, store=None):
    """Returns list of (Document, distance), most relevant first, irrelevant removed."""
    store = store or get_store()
    hits = store.similarity_search_with_score(question, k=k)
    return [(d, s) for d, s in hits if s <= MAX_DISTANCE]


def format_context(hits) -> str:
    return "\n\n".join(f"[{d.metadata['doc_id']} {d.metadata['title']}]\n{d.page_content}"
                       for d, _ in hits)


def sources_of(hits) -> list[dict]:
    seen, out = set(), []
    for d, s in hits:
        key = (d.metadata["doc_id"], d.metadata["section"])
        if key not in seen:
            seen.add(key)
            out.append({"doc_id": d.metadata["doc_id"], "title": d.metadata["title"],
                        "section": d.metadata["section"], "distance": round(float(s), 3)})
    return out


def ask(question: str, k: int = TOP_K, store=None, llm=None) -> dict:
    hits = retrieve(question, k, store)
    if not hits:
        return {"question": question, "answer": "I could not find this in the SOPs.", "sources": []}
    llm = llm or get_llm()
    msg = (PROMPT | llm).invoke({"context": format_context(hits), "question": question})
    return {"question": question, "answer": getattr(msg, "content", str(msg)).strip(),
            "sources": sources_of(hits)}


def print_result(r: dict):
    print("\n" + r["answer"])
    if r["sources"]:
        print("\nSources:")
        for s in r["sources"]:
            print(f"  - {s['doc_id']} {s['title']} > {s['section']}  (distance {s['distance']})")


# ------------------------------------------------------------------ CLI
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("question", nargs="*")
    ap.add_argument("--build", action="store_true", help="rebuild the vector index")
    args = ap.parse_args()

    emb = get_embeddings()
    if args.build or not CHROMA_DIR.exists():
        build_index(emb)
    store = get_store(emb)

    if args.question:
        print_result(ask(" ".join(args.question), store=store))
        return
    print("MacroOps SOP assistant. Type a question, or 'exit'.")
    llm = get_llm()
    while True:
        try:
            q = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if q.lower() in {"exit", "quit", ""}:
            break
        print_result(ask(q, store=store, llm=llm))


if __name__ == "__main__":
    sys.exit(main())