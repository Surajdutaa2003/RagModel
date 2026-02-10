Here is the **complete, self-contained README.md** file for your project — ready to copy-paste into the root of `RagFromScratch`.

```markdown
# RagFromScratch

Advanced **Agentic Retrieval-Augmented Generation (RAG)** pipeline built from scratch.

Combines hybrid retrieval (FAISS vector + BM25 keyword), LLM-powered self-query metadata filtering, RAG Fusion, Reciprocal Rank Fusion (RRF), large-model FlashRank reranking, long-context reordering, intelligent chunk compression, multi-store support, and clean source citations.

Designed for high-accuracy retrieval over private document collections with metadata.

## ✨ Features

- Multi-collection / multi-store retrieval (`company_store`, `movies`, `technical_store`, etc.)
- **Hybrid search** — dense vector (FAISS + Jina embeddings) + sparse keyword (BM25) via EnsembleRetriever
- **Self-query metadata filtering** — LLM parses natural questions into refined query + filters (`store`, `source`, `chunk_id`, etc.)
- **RAG Fusion** — generates multiple query variants for better recall
- **Reciprocal Rank Fusion (RRF)** — intelligently merges results from multiple retrievals
- **FlashRank reranking** — uses the largest listwise model (`rank_zephyr_7b_v1_full`) with cosine fallback
- **Lost-in-the-middle mitigation** — long-context reordering (most relevant chunks at start & end)
- **Chunk-level compression** — keeps only top relevant sentences per chunk using embedding similarity
- **Agentic workflow** — LangGraph agent decides when to call the RAG tool
- **Clean citations** — every returned chunk includes `[Source: … | chunk … | store …]`
- **Robust CLI** — interrupt-safe input loop (Ctrl+C friendly)
- Graceful fallback at every stage (retrieval, reranking, parsing)

## 📂 Project Structure

```
RagFromScratch/
├── src/
│   └── workflow.py           # Main agentic RAG script (CLI + LangGraph)
├── vector_store/
│   ├── company_store/        # FAISS index + meta.json
│   ├── movies/
│   └── technical_store/
├── .env                      # API keys (Azure OpenAI, Jina, etc.)
├── myenv/                    # Virtual environment
└── README.md
```

## 🚀 Quick Start

1. **Clone & activate environment**

```powershell
git clone <your-repo-url>
cd RagFromScratch
.\myenv\Scripts\Activate.ps1     # Windows PowerShell
# or source myenv/bin/activate    # Linux/macOS
```

2. **Install dependencies**

```bash
pip install langchain langchain-openai langchain-community langgraph faiss-cpu python-dotenv jina-openai rank_bm25 flashrank[listwise]
```

> Note: `flashrank[listwise]` is required for the large `rank_zephyr_7b_v1_full` model (~4 GB download on first use).

3. **Create .env file** in root

```env
# Azure OpenAI
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_API_KEY=sk-...
AZURE_OPENAI_DEPLOYMENT=gpt-5-chat
AZURE_OPENAI_API_VERSION=2025-01-01-preview

# Jina Embeddings
JINA_API_KEY=jina_...

# Optional: local quantized LLM
USE_QUANTIZED_LLM=0
QUANTIZED_LLM_PATH=/path/to/gguf/model.gguf
```

4. **Prepare vector stores**

Each store folder should contain:
- `faiss.index`
- `meta.json` with structure:
  ```json
  {
    "chunks": ["text chunk 1", "text chunk 2", ...],
    "metas": [
      {"source": "file.pdf", "chunk_id": 0, "store": "movies", "year": 1993, ...},
      ...
    ]
  }
  ```

5. **Run the agent**

```bash
python src/workflow.py
```

Example queries:
- "give me a sci-fi movie above 7 rating from 1993"
- "technical_store ke latest AI reports"
- "company_store mein chunk_id 10 se 20 wale sections"
- "animated movies from the 90s in movies store"

Type `exit`, `quit` or `q` to stop.

Ctrl+C is handled gracefully.

## 🛠️ Tech Stack

| Layer               | Tools / Libraries                                |
|---------------------|--------------------------------------------------|
| LLM                 | Azure OpenAI (gpt-5-chat) / Llama.cpp (optional) |
| Embeddings          | Jina Embeddings v2                               |
| Vector Store        | FAISS                                            |
| Keyword Search      | BM25 (rank_bm25)                                 |
| Hybrid Retrieval    | EnsembleRetriever                                |
| Self-Query          | Custom LLM parsing + post-filtering              |
| Reranking           | FlashRank (`rank_zephyr_7b_v1_full`)             |
| Fusion & Ranking    | RAG Fusion + Reciprocal Rank Fusion              |
| Agent Framework     | LangGraph                                        |
| Orchestration       | LangChain                                        |

## 🔍 How Self-Query Works

The system uses LLM to convert questions like:

> "sci-fi movies above 7 rating from 1993 in movies store"

into:

- refined query: `"sci-fi movies"`
- filters: `{"rating": ">7", "year": 1993, "store": "movies"}`

Filters are applied after retrieval (since FAISS lacks native metadata filtering).

## 📈 Roadmap / Future Ideas

- [ ] Add native support for more metadata fields (genre, author, rating, etc.)
- [ ] Switch to Chroma / Pinecone for native metadata filtering
- [ ] Add offline evaluation (hit rate, MRR, NDCG)
- [ ] Web UI (Streamlit / Gradio)
- [ ] Query rewriting / HyDE for even better recall
- [ ] Support for date ranges, numeric inequalities, contains operators

## ⚠️ Current Limitations

- FAISS does not support complex metadata filtering → post-filtering used (may discard some candidates)
- Large FlashRank model requires significant RAM (~4–8 GB) and `flashrank[listwise]` extra
- Self-query parsing quality depends on LLM prompt — complex questions may need prompt tuning
- No built-in query validation or error recovery for malformed metadata

 
