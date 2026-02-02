
import os
import json
import requests
import numpy as np
import faiss
from dotenv import load_dotenv 
from pathlib import Path
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# -----------------------------
# PDF Helper
# -----------------------------
def read_pdf(file_path: Path) -> str:
    reader = PdfReader(file_path)
    text = []
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text.append(page_text)
    return "\n".join(text)

# -----------------------------
# Load env
# -----------------------------
BASE_DIR = Path(__file__).parent.parent
ENV_PATH = BASE_DIR / ".env"
load_dotenv(dotenv_path=ENV_PATH)

JINA_API_KEY = os.getenv("JINA_API_KEY")
JINA_ENDPOINT = os.getenv(
    "JINA_EMBEDDINGS_ENDPOINT",
    "https://api.jina.ai/v1/embeddings"
)

# -----------------------------
# Paths
# -----------------------------
STORE_DIR = BASE_DIR / "vector_store"
INDEX_PATH = STORE_DIR / "faiss.index"
META_PATH = STORE_DIR / "meta.json"
DATA_DIR = BASE_DIR / "data"

# -----------------------------
# Chunking (now recursive)
# -----------------------------
def chunk_text(
    text: str,
    source: str,
    chunk_size: int = 500,
    chunk_overlap: int = 100,
):
    """
    Break text into meaningful chunks using recursive splitter + attach metadata
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ".", " ", ""],
        keep_separator=True
    )
    
    chunks = splitter.split_text(text)
    
    metas = []
    for i, chunk in enumerate(chunks):
        metas.append({
            "source": source,
            "chunk_id": i
        })
        
        
    
    return chunks, metas

# -----------------------------
# Call Jina Embeddings API
# -----------------------------
def get_embeddings(
    texts: list[str],
    model: str = "jina-embeddings-v2-base-en"
):
    headers = {
        "Authorization": f"Bearer {JINA_API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": model,
        "input": texts,
    }

    resp = requests.post(
        JINA_ENDPOINT,
        headers=headers,
        json=payload,
        timeout=60
    )

    if resp.status_code != 200:
        raise Exception(
            f"❌ Jina API Error {resp.status_code}: {resp.text}"
        )

    data = resp.json()
    embeddings = [item["embedding"] for item in data["data"]]
    return embeddings

# -----------------------------
# Build Vector Store
# -----------------------------
def build_vector_store():
    if not JINA_API_KEY:
        raise Exception("❌ Missing JINA_API_KEY in .env")

    STORE_DIR.mkdir(exist_ok=True)

    # 1️⃣ Load documents
    docs = list(DATA_DIR.glob("*.txt")) + list(DATA_DIR.glob("*.pdf"))
    if not docs:
        raise Exception(f"❌ No .txt or .pdf files found in {DATA_DIR}")

    all_chunks = []
    all_metas = []

    for doc_path in docs:
        if doc_path.suffix.lower() == ".pdf":
            text = read_pdf(doc_path)
        else:
            text = doc_path.read_text(encoding="utf-8")
        
        if not text.strip():
            print(f"⚠️ Skipping empty file: {doc_path.name}")
            continue

        chunks, metas = chunk_text(
            text=text,
            source=doc_path.name
        )

        all_chunks.extend(chunks)
        all_metas.extend(metas)

        print(f"   - Loaded: {doc_path.name} ({len(chunks)} chunks)")

    print(f"✅ Loaded {len(docs)} documents")
    print(f"✅ Total Chunks Created: {len(all_chunks)}")

    if not all_chunks:
        raise Exception("No chunks generated from documents")

    # 2️⃣ Generate embeddings (batched)
    all_embeddings = []
    batch_size = 16

    for i in range(0, len(all_chunks), batch_size):
        batch = all_chunks[i:i + batch_size]
        print(
            f"🔄 Embedding batch {i // batch_size + 1} / "
            f"{(len(all_chunks) - 1) // batch_size + 1}"
        )

        emb = get_embeddings(batch)
        all_embeddings.extend(emb)

    vectors = np.array(all_embeddings).astype("float32")
    dim = vectors.shape[1]
    print(f"✅ Embeddings shape: {vectors.shape}")

    # 3️⃣ Build FAISS index
    index = faiss.IndexFlatL2(dim)
    index.add(vectors)
    print(f"✅ FAISS index built. Total vectors: {index.ntotal}")

    # 4️⃣ Save FAISS index
    faiss.write_index(index, str(INDEX_PATH))
    print(f"💾 Saved FAISS index to: {INDEX_PATH}")

    # 5️⃣ Save metadata
    meta = {
        "chunks": all_chunks,
        "metas": all_metas
    }

    META_PATH.write_text(
        json.dumps(meta, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )
    print(f"💾 Saved metadata to: {META_PATH}")

    print("\n🎉 Vector store build complete!")

# -----------------------------
# Entry
# -----------------------------
if __name__ == "__main__":
    build_vector_store()