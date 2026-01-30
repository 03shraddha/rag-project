# RAG System From Scratch

A complete implementation of a Retrieval-Augmented Generation (RAG) system built from scratch using Python. This project demonstrates how to solve the hallucination problem in Large Language Models by grounding them with custom knowledge.

## What is RAG?

RAG (Retrieval-Augmented Generation) is a technique that gives AI models an "open-book exam" by:
1. Retrieving relevant information from your custom knowledge base
2. Augmenting the prompt with this context
3. Generating accurate, grounded answers

This prevents hallucinations and allows the model to answer questions about your specific data.

## Features

- **Custom Knowledge Base**: Query your own documents and data
- **No Hallucinations**: Model refuses to answer when information isn't available
- **100% Local**: All processing happens on your machine
- **Free & Open Source**: Uses free models and libraries
- **Fast Vector Search**: FAISS-powered semantic search

## Technology Stack

- **transformers** (Hugging Face): Free LLM (FLAN-T5-small)
- **sentence-transformers**: Embedding model (all-MiniLM-L6-v2)
- **faiss-cpu**: Facebook AI's vector search library
- **langchain**: Smart text chunking

## Installation

```bash
pip install transformers sentence-transformers faiss-cpu langchain langchain-text-splitters
```

## Usage

1. Add your knowledge to `my_knowledge.txt`
2. Run the RAG system:

```bash
python rag_system.py
```

## How It Works

### 1. Text Chunking
The knowledge base is split into overlapping chunks to maintain context:
```python
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=150,
    chunk_overlap=20
)
```

### 2. Embeddings
Each chunk is converted to a 384-dimensional vector using sentence-transformers:
```python
model = SentenceTransformer('all-MiniLM-L6-v2')
chunk_embeddings = model.encode(chunks)
```

### 3. Vector Store
FAISS indexes the embeddings for fast similarity search:
```python
index = faiss.IndexFlatL2(d)
index.add(chunk_embeddings)
```

### 4. Retrieval & Generation
When you ask a question:
- Your query is embedded
- Top-k similar chunks are retrieved
- Context is added to the prompt
- FLAN-T5 generates the answer

## Example Results

**Query**: "What is the WFH policy?"
**Answer**: "All employees are eligible for a hybrid WFH schedule"

**Query**: "What is the company's dental plan?" (not in knowledge base)
**Answer**: "I don't have that information." ✓ No hallucination!

## Project Structure

```
rag-project/
├── my_knowledge.txt      # Your custom knowledge base
├── rag_system.py         # Main RAG implementation
├── .gitignore
└── README.md
```

## Key Benefits

1. **Grounded Answers**: Model only uses your provided context
2. **No Hallucinations**: Refuses to answer when unsure
3. **Up-to-date Knowledge**: Just update the text file and re-run
4. **Privacy**: All data stays local on your machine
5. **Scalable**: Can handle large knowledge bases with FAISS

## Customization

- **Change the knowledge base**: Edit `my_knowledge.txt`
- **Adjust chunk size**: Modify `chunk_size` parameter
- **Retrieve more context**: Increase `k` value in search
- **Use different models**: Swap embedding or generation models

## License

MIT

## Acknowledgments

Built following best practices for production RAG systems using free, open-source tools.
