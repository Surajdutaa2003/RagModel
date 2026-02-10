"""
Agentic RAG Workflow with:
- FAISS retrieval
- LLM re-ranking replaced with Cross-Encoder (or embedding similarity fallback)a
- Hybrid retrieval (vector + BM25)
- Proper citations (file + chunk_id)
"""

import os
import json
from collections import defaultdict
from pathlib import Path
from typing import TypedDict, List, Annotated, Dict, Tuple
import numpy as np
import faiss
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, AIMessage, ToolMessage
from langchain_core.documents import Document
from langchain_core.tools import tool
from langchain_openai import AzureChatOpenAI
from langchain_community.embeddings import JinaEmbeddings
from langchain_community.retrievers.bm25 import BM25Retriever
from langchain_core.retrievers import BaseRetriever
import re
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

# FlashRank imports (added for replacement)
from langchain_community.document_compressors import FlashrankRerank

_FLASHRANK_READY = False


def _ensure_flashrank_ready() -> bool:
    """
    Ensure FlashRank Pydantic v2 models are fully defined.
    Returns True if ready, False otherwise.
    """
    global _FLASHRANK_READY
    if _FLASHRANK_READY:
        return True
    try:
        # Fix for Pydantic v2 "class-not-fully-defined" when Ranker isn't loaded
        from flashrank import Ranker  # type: ignore  # noqa: F401
        FlashrankRerank.model_rebuild()
        _FLASHRANK_READY = True
        return True
    except Exception:
        return False

# Remove sentence-transformers CrossEncoder (replaced by FlashRank)
# try:
#     from sentence_transformers import CrossEncoder
#     # Cross-Encoder for reranking (fast & accurate)
#     cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
# except Exception:
#     cross_encoder = None

# --------------------------------------------------
# Load env
# --------------------------------------------------
BASE_DIR = Path(__file__).parent.parent
load_dotenv(BASE_DIR / ".env")

# --------------------------------------------------
# Paths
# --------------------------------------------------
VECTOR_STORE_DIR = BASE_DIR / "vector_store"
MOVIES_METADATA_PATH = BASE_DIR / "data" / "movies_metadata.json"

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


def _build_movies_store(store_dir: Path) -> None:
    """
    Build a FAISS vector store from data/movies_metadata.json.
    """
    if not MOVIES_METADATA_PATH.exists():
        return

    raw = json.loads(MOVIES_METADATA_PATH.read_text(encoding="utf-8"))
    if not isinstance(raw, list) or not raw:
        return

    chunks: List[str] = []
    metas: List[dict] = []
    for i, item in enumerate(raw):
        if not isinstance(item, dict):
            continue
        text = item.get("page_content")
        meta = item.get("metadata") or {}
        if not isinstance(text, str) or not text.strip():
            continue
        if not isinstance(meta, dict):
            meta = {}
        meta = dict(meta)
        meta["source"] = "movies_metadata.json"
        meta["chunk_id"] = i
        chunks.append(text)
        metas.append(meta)

    if not chunks:
        return

    vectors = np.array(embeddings.embed_documents(chunks)).astype("float32")
    index = faiss.IndexFlatL2(vectors.shape[1])
    index.add(vectors)

    store_dir.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(store_dir / "faiss.index"))
    meta_payload = {"chunks": chunks, "metas": metas}
    (store_dir / "meta.json").write_text(json.dumps(meta_payload), encoding="utf-8")


STORE_DIRS = _discover_store_dirs()
if not STORE_DIRS:
    # Auto-build movies store if present
    movies_store_dir = VECTOR_STORE_DIR / "movies"
    _build_movies_store(movies_store_dir)
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
# Self-Query (LLM -> metadata filters)
# --------------------------------------------------
def _extract_json_block(text: str) -> str:
    if not text:
        return ""
    text = text.strip()
    if text.startswith("{") and text.endswith("}"):
        return text
    match = re.search(r"\{.*\}", text, re.DOTALL)
    return match.group(0) if match else ""


def self_query_parse(question: str, llm) -> Tuple[str, Dict[str, object]]:
    """
    Use LLM to convert question into a retrieval query + optional metadata filters.
    Returns (query, filters).
    """
    prompt = f"""You are a retrieval planner. Convert the user question into:
1) a concise retrieval query
2) optional metadata filters over fields: store, source, chunk_id

Return ONLY valid JSON with this schema:
{{
  "query": "string",
  "filters": {{
    "store": "string or null",
    "source": "string or null",
    "chunk_id": "int or null"
  }}
}}

Question: {question}
"""
    try:
        resp = llm.invoke(prompt)
        raw = getattr(resp, "content", "") or ""
        json_block = _extract_json_block(raw)
        if not json_block:
            return question, {}
        data = json.loads(json_block)
        query = data.get("query") or question
        filters = data.get("filters") or {}
        # normalize
        norm_filters = {}
        if isinstance(filters, dict):
            store = filters.get("store")
            source = filters.get("source")
            chunk_id = filters.get("chunk_id")
            if isinstance(store, str) and store.strip():
                norm_filters["store"] = store.strip()
            if isinstance(source, str) and source.strip():
                norm_filters["source"] = source.strip()
            if isinstance(chunk_id, int):
                norm_filters["chunk_id"] = chunk_id
            else:
                # allow numeric strings
                if isinstance(chunk_id, str) and chunk_id.strip().isdigit():
                    norm_filters["chunk_id"] = int(chunk_id.strip())
        return query, norm_filters
    except Exception:
        return question, {}


def _match_filters(meta: dict, filters: Dict[str, object]) -> bool:
    if not filters:
        return True
    store = filters.get("store")
    source = filters.get("source")
    chunk_id = filters.get("chunk_id")

    if store:
        if str(meta.get("store", "")).strip() != str(store).strip():
            return False
    if source:
        src = str(meta.get("source", "")).lower()
        if str(source).lower() not in src:
            return False
    if chunk_id is not None:
        try:
            if int(meta.get("chunk_id")) != int(chunk_id):
                return False
        except Exception:
            return False
    return True


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
# Re-ranking with FlashRank (replaces CrossEncoder; using Large model: ms-marco-MultiBERT-L-12)
# --------------------------------------------------
def rerank(question: str, chunks: List[dict], keep_top: int = 6):
    if not chunks:
        return []

    try:
        if not _ensure_flashrank_ready():
            raise RuntimeError("FlashRank is not available; install 'flashrank' to enable reranking")
        # Use FlashRank with largest available model (~4GB, listwise)
        reranker = FlashrankRerank(
            model="rank_zephyr_7b_v1_full",  # Largest model available
            top_n=keep_top
        )

        # Convert chunks to LangChain Documents
        langchain_docs = [
            Document(
                page_content=c["text"],
                metadata={
                    "source": c["source"],
                    "chunk_id": c["chunk_id"],
                    "store": c.get("store", "default"),
                    "score": 0.0  # placeholder
                }
            ) for c in chunks
        ]

        # Rerank and compress
        compressed_docs = reranker.compress_documents(
            documents=langchain_docs,
            query=question
        )

        # Convert back to dict format
        reranked = []
        for doc in compressed_docs:
            reranked.append({
                "text": doc.page_content,
                "source": doc.metadata["source"],
                "chunk_id": doc.metadata["chunk_id"],
                "store": doc.metadata.get("store", "default"),
                "score": doc.metadata.get("relevance_score", 0.0)
            })

        return reranked

    except Exception as e:
        print(f"FlashRank failed: {e} -> using embedding similarity fallback")
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


def generate_query_variants(question: str, llm, n: int = 5) -> List[str]:
    """
    RAG Fusion: generate multiple alternative queries for the same intent.
    """
    prompt = f"""Generate {n} different search queries that express the same intent
    as the question below. Keep them short and retrieval-focused.

    Question: {question}
    """

    resp = llm.invoke(prompt)
    lines = [
        l.strip("- ").strip()
        for l in resp.content.split("\n")
        if l.strip()
    ]

    # Always include original question
    return list(dict.fromkeys([question] + lines))


def reciprocal_rank_fusion(results: List[dict], k: int = 60) -> List[dict]:
    """
    Merge ranked results from multiple queries using RRF.
    """
    scores = defaultdict(float)
    doc_map = {}

    for r in results:
        # unique identity of a chunk across stores & queries
        doc_id = (
            r.get("store"),
            r.get("source"),
            r.get("chunk_id"),
        )

        rank = r.get("rank", 1000)
        scores[doc_id] += 1.0 / (rank + k)

        # Keep latest copy (text/metadata identical anyway)
        doc_map[doc_id] = r

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [doc_map[doc_id] for doc_id, _ in ranked]


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


def _run_retriever(retriever: BaseRetriever, query: str) -> List[Document]:
    """
    Compatibility wrapper across LangChain versions.
    """
    if hasattr(retriever, "get_relevant_documents"):
        return retriever.get_relevant_documents(query)
    return retriever.invoke(query)


# --------------------------------------------------
# TOOL: Agentic RAG Search with Hybrid + FlashRank (Large model)
# --------------------------------------------------
@tool
def rag_search(question: str) -> str:
    """
    RAG Fusion + Multi-store Hybrid Retrieval + RRF
    -> FlashRank (Large) -> Reorder -> Compression
    """
    # 0. Self-query: derive filters + refined query
    base_query, filters = self_query_parse(question, llm)

    # 1. Generate query variants (RAG Fusion)
    query_variants = generate_query_variants(base_query, llm)

    # 2. Prepare per-store hybrid retrievers
    if filters.get("store") and filters["store"] in STORES:
        store_names = [filters["store"]]
    else:
        store_names = list(STORES.keys())
    retrievers = [get_hybrid_retriever(name) for name in store_names]

    all_ranked_results = []

    # 3. Run retrieval for EACH query variant
    for q in query_variants:
        for store_name, retriever in zip(store_names, retrievers):
            docs = _run_retriever(retriever, q)

            # Fallback: if hybrid returns nothing, use direct vector retrieval
            if not docs:
                store = STORES[store_name]
                vec_results = retrieve(q, store["index"], store["chunks"], store["metas"], top_k=30)
                for r in vec_results:
                    if not _match_filters(r, filters):
                        continue
                    all_ranked_results.append({
                        "rank": r.get("rank", 1000),
                        "text": r["text"],
                        "source": r["source"],
                        "chunk_id": r["chunk_id"],
                        "store": r.get("store", store_name),
                    })
                continue

            for rank, d in enumerate(docs, start=1):
                meta = d.metadata
                if not _match_filters(meta, filters):
                    continue
                all_ranked_results.append({
                    "rank": rank,
                    "text": d.page_content,
                    "source": meta["source"],
                    "chunk_id": meta["chunk_id"],
                    "store": meta.get("store", store_name),
                })

    if not all_ranked_results:
        return "NO RELEVANT DOCUMENTS FOUND."

    # 4. RRF merge
    rrf_ranked = reciprocal_rank_fusion(all_ranked_results)

    # 5. FlashRank re-ranking (Large model)
    top = rerank(question, rrf_ranked)

    # 6. Context ordering + compression
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
    last_message = state["messages"][-1] if state["messages"] else None

    if isinstance(last_message, ToolMessage):
        # We already have tool output; now let the model answer.
        llm_with_tools = llm.bind_tools([rag_search])
    else:
        # Force tool call for new user questions.
        try:
            llm_with_tools = llm.bind_tools([rag_search], tool_choice="rag_search")
        except TypeError:
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
