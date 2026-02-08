import os
import time
import numpy as np
import faiss
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from transformers import T5Tokenizer, T5ForConditionalGeneration

# ── Global state ──────────────────────────────────────────────
rag = {}


def build_index():
    """Load knowledge, chunk, embed, and index."""
    print("[RAG] Loading knowledge base...")
    with open("my_knowledge.txt") as f:
        knowledge_text = f.read()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=150, chunk_overlap=20, length_function=len
    )
    chunks = splitter.split_text(knowledge_text)

    print("[RAG] Loading embedding model...")
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    chunk_embeddings = embed_model.encode(chunks).astype("float32")

    print("[RAG] Building FAISS index...")
    d = chunk_embeddings.shape[1]
    index = faiss.IndexFlatL2(d)
    index.add(chunk_embeddings)

    print("[RAG] Loading generative model...")
    tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-small")
    t5_model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-small")

    rag.update(
        {
            "knowledge_text": knowledge_text,
            "chunks": chunks,
            "chunk_embeddings": chunk_embeddings,
            "embed_model": embed_model,
            "index": index,
            "tokenizer": tokenizer,
            "t5_model": t5_model,
            "embedding_dim": d,
        }
    )
    print(f"[RAG] Ready — {len(chunks)} chunks, {d}-dim embeddings")


@asynccontextmanager
async def lifespan(app: FastAPI):
    build_index()
    yield


# ── App ───────────────────────────────────────────────────────
app = FastAPI(title="RAG Demo API", lifespan=lifespan)


class QueryRequest(BaseModel):
    question: str
    top_k: int = 2


class KnowledgeUpdate(BaseModel):
    content: str


# ── Routes ────────────────────────────────────────────────────


@app.get("/")
async def serve_ui():
    return FileResponse("static/index.html")


@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "chunks": len(rag.get("chunks", [])),
        "embedding_dim": rag.get("embedding_dim", 0),
        "index_size": rag["index"].ntotal if "index" in rag else 0,
    }


@app.get("/api/chunks")
async def get_chunks():
    """Return all chunks with their embeddings (first 8 dims for viz)."""
    chunks = rag["chunks"]
    embeddings = rag["chunk_embeddings"]
    return {
        "chunks": [
            {
                "id": i,
                "text": chunk,
                "embedding_preview": embeddings[i][:8].tolist(),
                "char_count": len(chunk),
            }
            for i, chunk in enumerate(chunks)
        ],
        "total": len(chunks),
        "embedding_dim": rag["embedding_dim"],
    }


@app.get("/api/knowledge")
async def get_knowledge():
    return {"content": rag["knowledge_text"]}


@app.post("/api/knowledge")
async def update_knowledge(body: KnowledgeUpdate):
    """Update knowledge base and rebuild index."""
    with open("my_knowledge.txt", "w") as f:
        f.write(body.content)
    build_index()
    return {"status": "rebuilt", "chunks": len(rag["chunks"])}


@app.post("/api/query")
async def query(body: QueryRequest):
    """Full RAG pipeline with timing and intermediate results."""
    question = body.question.strip()
    if not question:
        raise HTTPException(400, "Question cannot be empty")

    top_k = min(body.top_k, len(rag["chunks"]))
    steps = []

    # ── Step 1: Embed query ───────────────────────────────────
    t0 = time.perf_counter()
    query_embedding = rag["embed_model"].encode([question]).astype("float32")
    embed_ms = (time.perf_counter() - t0) * 1000
    steps.append(
        {
            "name": "Embed Query",
            "ms": round(embed_ms, 1),
            "detail": f"Encoded to {rag['embedding_dim']}-dim vector",
            "output": query_embedding[0][:8].tolist(),
        }
    )

    # ── Step 2: Retrieve ──────────────────────────────────────
    t0 = time.perf_counter()
    distances, indices = rag["index"].search(query_embedding, top_k)
    retrieve_ms = (time.perf_counter() - t0) * 1000

    retrieved = []
    for rank, (idx, dist) in enumerate(zip(indices[0], distances[0])):
        retrieved.append(
            {
                "rank": rank + 1,
                "chunk_id": int(idx),
                "text": rag["chunks"][idx],
                "distance": round(float(dist), 4),
                "similarity": round(1 / (1 + float(dist)), 4),
            }
        )

    steps.append(
        {
            "name": "Retrieve",
            "ms": round(retrieve_ms, 1),
            "detail": f"Top-{top_k} chunks via FAISS (L2 distance)",
            "output": retrieved,
        }
    )

    # ── Step 3: Augment ───────────────────────────────────────
    context = "\n\n".join(r["text"] for r in retrieved)
    prompt = f"""Answer the following question using *only* the provided context.
If the answer is not in the context, say "I don't have that information."

Context:
{context}

Question:
{question}

Answer:"""

    steps.append(
        {
            "name": "Augment",
            "ms": 0,
            "detail": "Built grounded prompt with retrieved context",
            "output": prompt,
        }
    )

    # ── Step 4: Generate ──────────────────────────────────────
    t0 = time.perf_counter()
    input_ids = rag["tokenizer"](
        prompt, return_tensors="pt", max_length=512, truncation=True
    ).input_ids
    outputs = rag["t5_model"].generate(input_ids, max_length=100)
    answer = rag["tokenizer"].decode(outputs[0], skip_special_tokens=True)
    gen_ms = (time.perf_counter() - t0) * 1000

    steps.append(
        {
            "name": "Generate",
            "ms": round(gen_ms, 1),
            "detail": "FLAN-T5-small inference",
            "output": answer,
        }
    )

    total_ms = sum(s["ms"] for s in steps)

    return {
        "question": question,
        "answer": answer,
        "steps": steps,
        "total_ms": round(total_ms, 1),
        "retrieved_chunks": retrieved,
    }


app.mount("/static", StaticFiles(directory="static"), name="static")
