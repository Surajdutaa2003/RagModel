"""
Agentic RAG Workflow with:
- FAISS retrieval
- LLM re-ranking → now replaced with Cross-Encoder (faster)
- Hybrid retrieval (vector + BM25)
- Proper citations (file + chunk_id)
"""

import os
import json
from pathlib import Path
from typing import TypedDict, List, Annotated

import numpy as np
import faiss
from dotenv import load_dotenv

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, AIMessage, Document
from langchain_core.tools import tool
from langchain_openai import AzureChatOpenAI
from langchain_community.embeddings import JinaEmbeddings
from langchain.retrievers import BM25Retriever, EnsembleRetriever

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from sentence_transformers import CrossEncoder

# --------------------------------------------------
# Load env
# --------------------------------------------------
BASE_DIR = Path(__file__).parent.parent
load_dotenv(BASE_DIR / ".env")

# --------------------------------------------------
# Paths
# --------------------------------------------------
FAISS_INDEX_PATH = BASE_DIR / "vector_store" / "faiss.index"
META_PATH = BASE_DIR / "vector_store" / "meta.json"

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

embeddings = JinaEmbeddings(
    jina_api_key=os.getenv("JINA_API_KEY"),
    model_name="jina-embeddings-v2-base-en",
)

# Cross-Encoder for reranking (fast & accurate)
cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

# --------------------------------------------------
# Load Vector Store
# --------------------------------------------------
def load_vector_store():
    index = faiss.read_index(str(FAISS_INDEX_PATH))
    meta = json.loads(META_PATH.read_text(encoding="utf-8"))
    return index, meta["chunks"], meta["metas"]

INDEX, CHUNKS, METAS = load_vector_store()

# --------------------------------------------------
# FAISS Retrieval (vector part)
# --------------------------------------------------
def retrieve(question: str, top_k: int = 10):
    q_embed = np.array([embeddings.embed_query(question)]).astype("float32")
    distances, indices = INDEX.search(q_embed, top_k)

    results = []
    for rank, idx in enumerate(indices[0], start=1):
        if idx == -1:
            continue
        meta = METAS[idx]
        results.append({
            "rank": rank,
            "text": CHUNKS[idx],
            "source": meta["source"],
            "chunk_id": meta["chunk_id"],
            "distance": float(distances[0][rank - 1]),
        })
    return results

# --------------------------------------------------
# Hybrid Retrieval (Vector + BM25)
# --------------------------------------------------
class FAISSRetriever:
    def get_relevant_documents(self, query: str):
        results = retrieve(query, top_k=10)
        return [Document(page_content=r["text"], metadata=r) for r in results]

def get_hybrid_retriever():
    vector_ret = FAISSRetriever()

    bm25_ret = BM25Retriever.from_texts(
        texts=CHUNKS,
        metadatas=METAS
    )
    bm25_ret.k = 10

    hybrid_ret = EnsembleRetriever(
        retrievers=[vector_ret, bm25_ret],
        weights=[0.4, 0.6]  # 40% vector, 60% keyword - tune as needed
    )
    return hybrid_ret

# --------------------------------------------------
# Re-ranking with Cross-Encoder
# --------------------------------------------------
def rerank(question: str, chunks: List[dict], keep_top: int = 3):
    if not chunks:
        return []

    pairs = [[question, c["text"]] for c in chunks]
    scores = cross_encoder.predict(pairs)

    # Sort by score descending
    sorted_pairs = sorted(zip(scores, chunks), key=lambda x: x[0], reverse=True)
    reranked = [chunk for _, chunk in sorted_pairs[:keep_top]]
    return reranked



# --------------------------------------------------
# TOOL: Agentic RAG Search with Hybrid + Cross-Encoder
# --------------------------------------------------
@tool
def rag_search(question: str) -> str:
    """
    Hybrid search (vector + BM25), rerank with Cross-Encoder, return formatted context.
    """
    hybrid_retriever = get_hybrid_retriever()
    docs = hybrid_retriever.get_relevant_documents(question)

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

    # Weak context check (optional - keep your threshold logic if you want)
    # avg_distance = sum(r["distance"] for r in top) / len(top) if top else 0
    # if avg_distance > 0.65:
    #     # refinement logic...

    blocks = []
    for r in top:
        blocks.append(
            f"[Source: {r['source']} | chunk {r['chunk_id']}]\n"
            f"{r['text'].strip()}"
        )

    return "\n\n──────────\n\n".join(blocks)

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
    print("\n🧠 Agentic RAG Ready (type 'exit', 'quit' or 'q' to quit)\n")

    system_prompt = SystemMessage(content="""
You are a professional research assistant with access to a specific document collection.

Rules you MUST follow:
- Use the rag_search tool for every factual question or when you need information
- Base your answer ONLY on the retrieved context — never hallucinate or use outside knowledge
- Use clear bullet points when appropriate
- Always include clear citations to sources and chunk ids
- If the retrieved documents do not contain the answer, reply only: "I don't have sufficient information from the provided documents to answer this."
- Be concise and accurate
""")

    while True:
        query = input("👤 You: ").strip()
        if query.lower() in ["exit", "quit", "q"]:
            print("Goodbye!")
            break

        initial_state = {
            "messages": [
                system_prompt,
                HumanMessage(content=query),
            ]
        }

        try:
            result = graph.invoke(initial_state)
            final_message = result["messages"][-1]
            print("\n🤖 Answer:\n")
            print(final_message.content)
            print("\n" + "═" * 70 + "\n")
        except Exception as e:
            print(f"\nError during execution: {e}\n")