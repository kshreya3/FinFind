# FinFind — AI Financial Document Chatbot

![Python](https://img.shields.io/badge/python-3.11%2B-blue)
![Gemini](https://img.shields.io/badge/AI-Gemini%20API-4285F4)
![RAG](https://img.shields.io/badge/architecture-RAG-orange)
![License](https://img.shields.io/badge/license-MIT-green)

Conversational AI for financial document analysis — upload 10-K filings, earnings reports, or loan agreements, then ask questions in plain English. Combines **RAG (Retrieval-Augmented Generation)** with live internet search for questions requiring current market context.

---

## Architecture

```
  User Question
       │
       ▼
  ┌───────────────────────────────────────────────┐
  │                 FinFind Pipeline               │
  │                                               │
  │  Query Classifier                             │
  │  ├── Document-grounded? → RAG pipeline        │
  │  └── Requires current data? → Search pipeline │
  │                                               │
  │  ┌─────────── RAG Pipeline ──────────────┐    │
  │  │  PDF Parser (PyMuPDF)                 │    │
  │  │    └── chunk text (512 tokens, 64     │    │
  │  │         overlap) → embed (Gemini      │    │
  │  │         Embedding) → Pinecone store   │    │
  │  │                                       │    │
  │  │  Retriever                            │    │
  │  │    └── similarity search → top-k=5   │    │
  │  │         chunks → reranker            │    │
  │  │                                       │    │
  │  │  Generator (Gemini API)               │    │
  │  │    └── [context chunks] + question   │    │
  │  │         → grounded answer + citation  │    │
  │  └───────────────────────────────────────┘    │
  │                                               │
  │  ┌────────── Search Pipeline ────────────┐    │
  │  │  Google Search API → top 5 results    │    │
  │  │    └── extract snippets + URLs        │    │
  │  │  Gemini API → synthesize + cite       │    │
  │  └───────────────────────────────────────┘    │
  │                                               │
  │  Response with citations + source type        │
  └───────────────────────────────────────────────┘
```

---

## Key Features

| Feature | Detail |
|---|---|
| **Document ingestion** | PDF, DOCX, TXT — up to 500 pages |
| **Chunking strategy** | 512-token chunks, 64-token overlap, sentence-boundary aware |
| **Embeddings** | Gemini Embedding API (768-dim) |
| **Vector store** | Pinecone (serverless) |
| **Hybrid retrieval** | Semantic search + keyword fallback |
| **Reranking** | Cross-encoder reranker for top-k chunk selection |
| **Internet search** | Google Custom Search API for live market data |
| **Citation tracking** | Every answer includes page number + source URL |
| **Multi-document** | Query across multiple uploaded documents simultaneously |

---

## Quick Start

```bash
git clone https://github.com/kshreya3/FinFind
cd finnfind
pip install -r requirements.txt

# Configure
cp .env.example .env
# Add: GEMINI_API_KEY, PINECONE_API_KEY, GOOGLE_SEARCH_API_KEY

# Start
uvicorn app.server:app --port 8000
```

Open `http://localhost:8000` → Upload a financial PDF → Start chatting.

---

## Example Interactions

```
User: What was Tesla's gross margin in Q3 2024?
FinFind: Tesla's gross margin in Q3 2024 was 17.1%, down from 19.8% in Q3 2023,
         driven by continued price reductions on Model 3 and Model Y.
         [Source: 10-Q filing, page 28 — uploaded document]

User: How does that compare to the current industry average?
FinFind: As of Q3 2024, the auto industry gross margin average is approximately
         14.2% (Ford: 10.1%, GM: 12.8%, Rivian: -44.3%).
         Tesla remains above industry average despite margin compression.
         [Source: Google Search — earnings reports, Oct 2024]
```

The system automatically selects RAG (document-grounded) or search (live data) based on whether the question requires uploaded content or current market information.

---

## RAG Implementation Details

### Chunking

```python
def chunk_document(text: str, chunk_size: int = 512, overlap: int = 64) -> list[Chunk]:
    # Split on sentence boundaries first, then enforce token limit
    sentences = sent_tokenize(text)
    chunks = []
    current_chunk = []
    current_tokens = 0

    for sentence in sentences:
        tokens = len(sentence.split())
        if current_tokens + tokens > chunk_size:
            chunks.append(Chunk(text=" ".join(current_chunk)))
            # Keep overlap window
            current_chunk = current_chunk[-overlap_sentences:]
            current_tokens = sum(len(s.split()) for s in current_chunk)
        current_chunk.append(sentence)
        current_tokens += tokens

    return chunks
```

### Retrieval + Reranking

```python
async def retrieve(query: str, top_k: int = 5) -> list[Chunk]:
    # 1. Embed query
    query_embedding = await gemini_embed(query)

    # 2. Semantic search in Pinecone
    candidates = pinecone_index.query(
        vector=query_embedding,
        top_k=top_k * 3,  # over-fetch for reranker
        include_metadata=True
    )

    # 3. Rerank with cross-encoder
    reranked = cross_encoder.rank(query, [c.text for c in candidates])
    return reranked[:top_k]
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| AI / LLM | Gemini API (gemini-2.0-flash) |
| Embeddings | Gemini Embedding API |
| Vector store | Pinecone (serverless) |
| PDF parsing | PyMuPDF |
| Reranking | sentence-transformers cross-encoder |
| Web search | Google Custom Search API |
| API server | FastAPI |
| Frontend | React + shadcn/ui |

---

## License

MIT
