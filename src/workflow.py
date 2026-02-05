"""
Agentic RAG Workflow with:
- FAISS retrieval
- LLM re-ranking -> now replaced with Cross-Encoder (faster)
- Hybrid retrieval (vector + BM25)
- Proper citations (file + chunk_id)
"""

import os
import json
from pathlib import Path
from typing import TypedDict, List, Annotated, Dict, Tuple

import numpy as np
import faiss
from dotenv import load_dotenv

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, AIMessage
from langchain_core.documents import Document
from langchain_core.tools import tool
from langchain_openai import AzureChatOpenAI
from langchain_community.embeddings import JinaEmbeddings
from langchain_community.retrievers.bm25 import BM25Retriever
from langchain_core.retrievers import BaseRetriever
try:
    # Pydantic v2
    from pydantic import ConfigDict
except Exception:
    ConfigDict = None
try:
    # Newer langchain package
    from langchain.retrievers import EnsembleRetriever
except Exception:
    try:
        # Older "classic" split
        from langchain_classic.retrievers import EnsembleRetriever
    except Exception:
        # If only langchain-community is installed
        from langchain_community.retrievers import EnsembleRetriever

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
try:
    from langchain_community.llms import LlamaCpp
except Exception:
    LlamaCpp = None
try:
    from sentence_transformers import CrossEncoder
    # Cross-Encoder for reranking (fast & accurate)
    cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
except Exception:
    cross_encoder = None

# --------------------------------------------------
# Load env
# --------------------------------------------------
BASE_DIR = Path(__file__).parent.parent
load_dotenv(BASE_DIR / ".env")

# --------------------------------------------------
# Paths
# --------------------------------------------------
VECTOR_STORE_DIR = BASE_DIR / "vector_store"

# --------------------------------------------------
# Models
# --------------------------------------------------
llm = AzureChatOpenAI(
    azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    temperature=0.0,
)
USE_QUANTIZED_LLM = os.getenv("USE_QUANTIZED_LLM", "0") == "1"
QUANTIZED_LLM_PATH = os.getenv("QUANTIZED_LLM_PATH", "").strip()

if USE_QUANTIZED_LLM and QUANTIZED_LLM_PATH and LlamaCpp is not None:
    answer_llm = LlamaCpp(
        model_path=QUANTIZED_LLM_PATH,
        temperature=0.0,
        n_ctx=4096,
    )
else:
    answer_llm = llm

embeddings = JinaEmbeddings(
    jina_api_key=os.getenv("JINA_API_KEY"),
    model_name="jina-embeddings-v2-base-en",
)


# --------------------------------------------------
# Load Vector Store
# --------------------------------------------------
def _discover_store_dirs() -> List[Path]:
    if VECTOR_STORE_DIR.exists():
        dirs = [
            p for p in VECTOR_STORE_DIR.iterdir()
            if p.is_dir() and (p / "faiss.index").exists() and (p / "meta.json").exists()
        ]
        if dirs:
            return dirs

    # Fallback to single-store layout at vector_store/*
    if (VECTOR_STORE_DIR / "faiss.index").exists() and (VECTOR_STORE_DIR / "meta.json").exists():
        return [VECTOR_STORE_DIR]

    return []


def load_vector_store(store_dir: Path) -> Tuple[faiss.Index, List[str], List[dict]]:
    index = faiss.read_index(str(store_dir / "faiss.index"))
    meta = json.loads((store_dir / "meta.json").read_text(encoding="utf-8"))
    return index, meta["chunks"], meta["metas"]


STORE_DIRS = _discover_store_dirs()
if not STORE_DIRS:
    raise FileNotFoundError(
        "No vector stores found. Expected vector_store/<store>/faiss.index and meta.json"
    )

STORES: Dict[str, Dict[str, object]] = {}
for store_dir in STORE_DIRS:
    name = store_dir.name
    index, chunks, metas = load_vector_store(store_dir)
    # Stamp store name into metadata for citations/merging
    stamped_metas = []
    for m in metas:
        stamped = dict(m)
        stamped["store"] = name
        stamped_metas.append(stamped)
    STORES[name] = {
        "index": index,
        "chunks": chunks,
        "metas": stamped_metas,
    }

# --------------------------------------------------
# FAISS Retrieval (vector part)
# --------------------------------------------------
def retrieve(question: str, index, chunks, metas, top_k: int = 10):
    q_embed = np.array([embeddings.embed_query(question)]).astype("float32")
    distances, indices = index.search(q_embed, top_k)

    results = []
    for rank, idx in enumerate(indices[0], start=1):
        if idx == -1:
            continue
        meta = metas[idx]
        results.append({
            "rank": rank,
            "text": chunks[idx],
            "source": meta["source"],
            "chunk_id": meta["chunk_id"],
            "store": meta.get("store"),
            "distance": float(distances[0][rank - 1]),
        })
    return results


# --------------------------------------------------
# Hybrid Retrieval (Vector + BM25)
# --------------------------------------------------
class FAISSRetriever(BaseRetriever):
    if ConfigDict is not None:
        model_config = ConfigDict(arbitrary_types_allowed=True)

    index: object
    chunks: List[str]
    metas: List[dict]
    store_name: str

    def _get_relevant_documents(self, query: str):
        results = retrieve(query, self.index, self.chunks, self.metas, top_k=10)
        return [Document(page_content=r["text"], metadata=r) for r in results]


def get_hybrid_retriever(store_name: str):
    store = STORES[store_name]
    vector_ret = FAISSRetriever(
        index=store["index"],
        chunks=store["chunks"],
        metas=store["metas"],
        store_name=store_name,
    )

    try:
        bm25_ret = BM25Retriever.from_texts(
            texts=store["chunks"],
            metadatas=store["metas"]
        )
        bm25_ret.k = 10

        hybrid_ret = EnsembleRetriever(
            retrievers=[vector_ret, bm25_ret],
            weights=[0.4, 0.6]  # 40% vector, 60% keyword - tune as needed
        )
        print(f"BM25 Hybrid Retriever initialized for {store_name} (Vector 40% + BM25 60%)")
        return hybrid_ret
    except Exception as e:
        # BM25 (rank_bm25) not available - fall back to vector-only retriever
        print(f"BM25 not available for {store_name}; using vector-only retriever (install 'rank_bm25' to enable BM25).")
        print(f"Error: {e}")
        return vector_ret


class MultiStoreRetriever(BaseRetriever):
    if ConfigDict is not None:
        model_config = ConfigDict(arbitrary_types_allowed=True)

    retrievers: List[BaseRetriever]

    def _get_relevant_documents(self, query: str):
        docs = []
        for r in self.retrievers:
            docs.extend(r.invoke(query))
        # Deduplicate by store/source/chunk_id
        seen = set()
        deduped = []
        for d in docs:
            key = (
                d.metadata.get("store"),
                d.metadata.get("source"),
                d.metadata.get("chunk_id"),
            )
            if key in seen:
                continue
            seen.add(key)
            deduped.append(d)
        return deduped


def get_merger_retriever():
    per_store = [get_hybrid_retriever(name) for name in STORES.keys()]
    return MultiStoreRetriever(retrievers=per_store)


# --------------------------------------------------
# Re-ranking with Cross-Encoder (or embeddings fallback)
# --------------------------------------------------
def rerank(question: str, chunks: List[dict], keep_top: int = 6):
    if not chunks:
        return []

    if cross_encoder is not None:
        pairs = [[question, c["text"]] for c in chunks]
        scores = cross_encoder.predict(pairs)
    else:
        # Fallback: use embedding cosine similarity between question and each chunk
        q_emb = np.array(embeddings.embed_query(question)).astype("float32")
        docs_emb = np.array(embeddings.embed_documents([c["text"] for c in chunks])).astype("float32")
        # normalize
        q_norm = q_emb / (np.linalg.norm(q_emb) + 1e-12)
        d_norm = docs_emb / (np.linalg.norm(docs_emb, axis=1, keepdims=True) + 1e-12)
        scores = (d_norm @ q_norm).tolist()

    # Sort by score descending
    sorted_pairs = sorted(zip(scores, chunks), key=lambda x: x[0], reverse=True)
    reranked = []
    for score, chunk in sorted_pairs[:keep_top]:
        enriched = dict(chunk)
        enriched["score"] = float(score)
        reranked.append(enriched)
    return reranked


def long_context_reorder(chunks: List[dict]) -> List[dict]:
    """
    Place the most relevant chunks at the beginning and end to reduce
    lost-in-the-middle effects.
    """
    if not chunks:
        return []

    left = []
    right = []
    for i, ch in enumerate(chunks):
        if i % 2 == 0:
            left.append(ch)
        else:
            right.append(ch)
    return left + list(reversed(right))


def _split_sentences(text: str) -> List[str]:
    # Lightweight splitter to avoid extra deps
    lines = [l.strip() for l in text.replace("\n", " ").split(". ") if l.strip()]
    sentences = []
    for l in lines:
        if not l.endswith("."):
            l = l + "."
        sentences.append(l)
    return sentences


def compress_chunks(question: str, chunks: List[dict], max_sentences_per_chunk: int = 3) -> List[dict]:
    if not chunks:
        return []

    q_emb = np.array(embeddings.embed_query(question)).astype("float32")
    q_norm = q_emb / (np.linalg.norm(q_emb) + 1e-12)

    compressed = []
    for ch in chunks:
        sentences = _split_sentences(ch["text"])
        if not sentences:
            compressed.append(ch)
            continue
        s_emb = np.array(embeddings.embed_documents(sentences)).astype("float32")
        s_norm = s_emb / (np.linalg.norm(s_emb, axis=1, keepdims=True) + 1e-12)
        scores = (s_norm @ q_norm).tolist()

        # Select top sentences by score, keep original order
        top_idx = sorted(
            sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:max_sentences_per_chunk]
        )
        kept = " ".join([sentences[i] for i in top_idx]).strip()
        new_chunk = dict(ch)
        new_chunk["text"] = kept if kept else ch["text"]
        compressed.append(new_chunk)
    return compressed


# --------------------------------------------------
# TOOL: Agentic RAG Search with Hybrid + Cross-Encoder
# --------------------------------------------------
@tool
def rag_search(question: str) -> str:
    """
    Multi-store retrieval -> Merger -> Re-rank -> Long-context reorder -> Compression.
    """
    merger = get_merger_retriever()
    docs = merger.invoke(question)

    chunks = [
        {
            "text": d.page_content,
            "source": d.metadata["source"],
            "chunk_id": d.metadata["chunk_id"],
            "distance": d.metadata.get("distance", 0.0)  # may not always exist from BM25
        }
        for d in docs
    ]

    if not chunks:
        return "NO RELEVANT DOCUMENTS FOUND."

    # Rerank with Cross-Encoder
    top = rerank(question, chunks)
    reordered = long_context_reorder(top)
    compressed = compress_chunks(question, reordered)

    blocks = []
    for r in compressed:
        blocks.append(
            f"[Source: {r['source']} | chunk {r['chunk_id']} | store {r.get('store', 'default')}]\n"
            f"{r['text'].strip()}"
        )

    return "\n\n----------\n\n".join(blocks)


# --------------------------------------------------
# State
# --------------------------------------------------
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]


# --------------------------------------------------
# Agent Node
# --------------------------------------------------
def agent_node(state: AgentState) -> dict:
    llm_with_tools = llm.bind_tools([rag_search])

    response: AIMessage = llm_with_tools.invoke(state["messages"])
    response = _normalize_tool_call(response)
    return {"messages": [response]}


# --------------------------------------------------
# Router
# --------------------------------------------------
def should_continue(state: AgentState) -> str:
    last_message = state["messages"][-1]
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        return "tools"
    return END


# --------------------------------------------------
# Fallback: parse printed tool call JSON (if model doesn't emit tool_calls)
# --------------------------------------------------
def _normalize_tool_call(message: AIMessage) -> AIMessage:
    if message.tool_calls:
        return message

    content = (message.content or "").strip()
    if not content.startswith("{"):
        return message

    first_line = content.splitlines()[0].strip()
    try:
        data = json.loads(first_line)
    except Exception:
        return message

    name = data.get("name")
    args = data.get("arguments")
    if name in ("rag_search", "functions.rag_search") and isinstance(args, dict):
        tool_call = {
            "name": "rag_search",
            "args": args,
            "id": "manual-rag-search",
            "type": "tool_call",
        }
        # Replace content to avoid the model's placeholder answer
        return AIMessage(content="", tool_calls=[tool_call])

    return message


# --------------------------------------------------
# Build Graph
# --------------------------------------------------
workflow = StateGraph(AgentState)

workflow.add_node("agent", agent_node)
workflow.add_node("tools", ToolNode([rag_search]))

workflow.add_edge(START, "agent")
workflow.add_conditional_edges(
    "agent",
    should_continue,
    {"tools": "tools", END: END}
)
workflow.add_edge("tools", "agent")

graph = workflow.compile()


# --------------------------------------------------
# Run (CLI interface)
# --------------------------------------------------
if __name__ == "__main__":
    print("\nAgentic RAG Ready (type 'exit', 'quit' or 'q' to quit)\n")

    system_prompt = SystemMessage(content="""
You are a professional research assistant with access to a specific document collection.

Rules you MUST follow:
- Use the rag_search tool for every factual question or when you need information
- Base your answer ONLY on the retrieved context - never hallucinate or use outside knowledge
- Use clear bullet points when appropriate
- Always include clear citations to sources and chunk ids
- If the retrieved documents do not contain the answer, reply only: "I don't have sufficient information from the provided documents to answer this."
- Be concise and accurate
""")

    DIRECT_PIPELINE = USE_QUANTIZED_LLM

    while True:
        query = input("You: ").strip()
        if query.lower() in ["exit", "quit", "q"]:
            print("Goodbye!")
            break

        if DIRECT_PIPELINE:
            try:
                context = rag_search(query)
                prompt = (
                    "Answer the question using ONLY the context below. "
                    "Always include citations with [Source: ... | chunk ... | store ...].\n\n"
                    f"Question: {query}\n\n"
                    f"Context:\n{context}\n"
                )
                if hasattr(answer_llm, "invoke"):
                    answer = answer_llm.invoke(prompt)
                    content = answer if isinstance(answer, str) else getattr(answer, "content", str(answer))
                else:
                    content = str(answer_llm(prompt))
                print("\nAnswer:\n")
                print(content)
                print("\n" + "=" * 70 + "\n")
            except Exception as e:
                print(f"\nError during execution: {e}\n")
        else:
            initial_state = {
                "messages": [
                    system_prompt,
                    HumanMessage(content=query),
                ]
            }

            try:
                result = graph.invoke(initial_state)
                final_message = result["messages"][-1]
                print("\nAnswer:\n")
                print(final_message.content)
                print("\n" + "=" * 70 + "\n")
            except Exception as e:
                print(f"\nError during execution: {e}\n")
