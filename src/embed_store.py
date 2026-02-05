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
# Paths (now per category)
# -----------------------------
STORE_DIR = BASE_DIR / "vector_store"
STORE_DIR.mkdir(exist_ok=True)
DATA_DIR = BASE_DIR / "data"

# -----------------------------
# Chunking (recursive)
# -----------------------------
def chunk_text(
    text: str,
    source: str,
    chunk_size: int = 500,
    chunk_overlap: int = 100,
):
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
# Build Multiple Vector Stores
# -----------------------------
def build_vector_store():
    if not JINA_API_KEY:
        raise Exception("❌ Missing JINA_API_KEY in .env")

    # Create sub-folders for each store
    company_dir = STORE_DIR / "company_store"
    technical_dir = STORE_DIR / "technical_store"
    company_dir.mkdir(exist_ok=True)
    technical_dir.mkdir(exist_ok=True)

    # Company store files
    company_index_path = company_dir / "faiss.index"
    company_meta_path = company_dir / "meta.json"

    # Technical store files
    technical_index_path = technical_dir / "faiss.index"
    technical_meta_path = technical_dir / "meta.json"

    # Load all files
    docs = list(DATA_DIR.glob("*.txt")) + list(DATA_DIR.glob("*.pdf"))
    if not docs:
        raise Exception(f"❌ No .txt or .pdf files found in {DATA_DIR}")

    # Separate into categories
    company_chunks = []
    company_metas = []
    technical_chunks = []
    technical_metas = []

    for doc_path in docs:
        filename = doc_path.name.lower()
        if "company" in filename or "faq" in filename:
            # Company category
            text = read_pdf(doc_path) if doc_path.suffix.lower() == ".pdf" else doc_path.read_text(encoding="utf-8")
            if not text.strip():
                print(f"⚠️ Skipping empty company file: {doc_path.name}")
                continue
            chunks, metas = chunk_text(text, doc_path.name)
            company_chunks.extend(chunks)
            company_metas.extend(metas)
            print(f"   - Company: {doc_path.name} ({len(chunks)} chunks)")
        else:
            # Technical category (default for all others)
            text = read_pdf(doc_path) if doc_path.suffix.lower() == ".pdf" else doc_path.read_text(encoding="utf-8")
            if not text.strip():
                print(f"⚠️ Skipping empty technical file: {doc_path.name}")
                continue
            chunks, metas = chunk_text(text, doc_path.name)
            technical_chunks.extend(chunks)
            technical_metas.extend(metas)
            print(f"   - Technical: {doc_path.name} ({len(chunks)} chunks)")

    print(f"✅ Company chunks: {len(company_chunks)}")
    print(f"✅ Technical chunks: {len(technical_chunks)}")

    # Build company store
    if company_chunks:
        company_vectors = np.array(get_embeddings(company_chunks)).astype("float32")
        company_dim = company_vectors.shape[1]
        company_index = faiss.IndexFlatL2(company_dim)
        company_index.add(company_vectors)
        faiss.write_index(company_index, str(company_index_path))
        meta_company = {"chunks": company_chunks, "metas": company_metas}
        company_meta_path.write_text(
            json.dumps(meta_company, indent=2, ensure_ascii=False),
            encoding="utf-8"
        )
        print(f"💾 Saved company store: {company_index_path}")

    # Build technical store
    if technical_chunks:
        technical_vectors = np.array(get_embeddings(technical_chunks)).astype("float32")
        technical_dim = technical_vectors.shape[1]
        technical_index = faiss.IndexFlatL2(technical_dim)
        technical_index.add(technical_vectors)
        faiss.write_index(technical_index, str(technical_index_path))
        meta_technical = {"chunks": technical_chunks, "metas": technical_metas}
        technical_meta_path.write_text(
            json.dumps(meta_technical, indent=2, ensure_ascii=False),
            encoding="utf-8"
        )
        print(f"💾 Saved technical store: {technical_index_path}")

    print("\n🎉 Multiple vector stores build complete!")

# -----------------------------
# Entry
# -----------------------------
if __name__ == "__main__":
    build_vector_store()
