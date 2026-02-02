# src/chat.py

import os
import json
import numpy as np
import faiss
from dotenv import load_dotenv
from pathlib import Path

from langchain_openai import AzureChatOpenAI
from langchain_community.embeddings import JinaEmbeddings

# -----------------------------
# Load .env properly
# -----------------------------
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

# -----------------------------
# Paths
# -----------------------------
BASE_DIR = Path(__file__).parent.parent
FAISS_INDEX_PATH = BASE_DIR / "vector_store" / "faiss.index"
META_PATH = BASE_DIR / "vector_store" / "meta.json"

# -----------------------------
# Models
# -----------------------------
llm = AzureChatOpenAI(
    azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
)

embeddings = JinaEmbeddings(
    jina_api_key=os.getenv("JINA_API_KEY"),
    model_name="jina-embeddings-v2-base-en"
)


# -----------------------------
# Load FAISS + chunks + metas
# -----------------------------
def load_vector_store():
    if not FAISS_INDEX_PATH.exists():
        raise FileNotFoundError(f"FAISS index not found: {FAISS_INDEX_PATH}")

    if not META_PATH.exists():
        raise FileNotFoundError(f"Metadata file not found: {META_PATH}")

    index = faiss.read_index(str(FAISS_INDEX_PATH))

    with open(META_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)

    chunks = meta["chunks"]

    # ✅ NEW: metas (source info)
    metas = meta.get("metas", [])

    return index, chunks, metas


# -----------------------------
# Retrieve top-k chunks
# -----------------------------
def retrieve_chunks(question: str, index, chunks, metas, top_k: int = 3):
    q_embed = embeddings.embed_query(question)
    q_embed = np.array([q_embed]).astype("float32")

    distances, indices = index.search(q_embed, top_k)

    results = []
    for rank, idx in enumerate(indices[0], start=1):
        if idx == -1:
            continue

        chunk_text = chunks[idx]

        # ✅ meta safe access
        if metas and idx < len(metas):
            source = metas[idx].get("source", "unknown")
            chunk_id = metas[idx].get("chunk_id", idx)
        else:
            source = "unknown"
            chunk_id = idx

        results.append({
            "rank": rank,
            "chunk_index": idx,
            "chunk_id": chunk_id,
            "source": source,
            "distance": float(distances[0][rank - 1]),
            "text": chunk_text
        })

    return results


# -----------------------------
# Ask LLM using retrieved context
# -----------------------------
def answer_question(question: str, retrieved_chunks):
    # include sources in context
    context_blocks = []
    for ch in retrieved_chunks:
        context_blocks.append(
            f"[SOURCE: {ch['source']} | CHUNK_ID: {ch['chunk_id']}]\n{ch['text']}"
        )

    context = "\n\n---\n\n".join(context_blocks)

    prompt = f"""
You are a helpful RAG assistant.
Answer strictly from the given context.
If the answer is not present in the context, say: "I don't know from the given documents."

Context:
{context}

User Question:
{question}

Answer in simple Hinglish.
Rules:
- Give bullet points
- Mention sources like: (source: <file> | chunk_id: <id>)
"""

    response = llm.invoke(prompt).content
    return response


# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    print("✅ Loading FAISS vector store...")
    index, chunks, metas = load_vector_store()

    print("\n🧠 RAG Chat Ready! Type your question. (exit/quit/q to stop)\n")

    while True:
        question = input("👤 You: ").strip()

        if question.lower() in ["exit", "quit", "q"]:
            print("👋 Bye bro!")
            break

        retrieved = retrieve_chunks(question, index, chunks, metas, top_k=3)

        print("\n📌 Retrieved Context Chunks:\n")
        for item in retrieved:
            print(f"--- Rank {item['rank']} ---")
            print(f"Source: {item['source']} | ChunkID: {item['chunk_id']} | Distance: {item['distance']:.4f}")
            print(item["text"][:300] + "...\n")

        final_answer = answer_question(question, retrieved)

        print("\n🤖 Answer:\n")
        print(final_answer)
        print("\n" + "=" * 60 + "\n")
  