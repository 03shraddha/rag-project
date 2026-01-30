import os
import numpy as np
import faiss
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from transformers import T5Tokenizer, T5ForConditionalGeneration

print("=" * 60)
print("Building RAG System From Scratch")
print("=" * 60)

# Step 1: Load the knowledge base
print("\n[Step 1] Loading knowledge base...")
with open("my_knowledge.txt") as f:
    knowledge_text = f.read()
print("Knowledge base loaded successfully!")

# Step 2: Chunking with LangChain
print("\n[Step 2] Chunking the text...")
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=150,
    chunk_overlap=20,
    length_function=len
)

chunks = text_splitter.split_text(knowledge_text)
print(f"Created {len(chunks)} chunks:")
for i, chunk in enumerate(chunks):
    print(f"--- Chunk {i+1} ---\n{chunk}\n")

# Step 3: Create embeddings
print("\n[Step 3] Creating embeddings...")
print("Loading embedding model 'all-MiniLM-L6-v2'...")
model = SentenceTransformer('all-MiniLM-L6-v2')

print("Encoding chunks...")
chunk_embeddings = model.encode(chunks)
print(f"Shape of our embeddings: {chunk_embeddings.shape}")

# Step 4: Create FAISS vector store
print("\n[Step 4] Setting up FAISS vector store...")
d = chunk_embeddings.shape[1]
index = faiss.IndexFlatL2(d)
index.add(np.array(chunk_embeddings).astype('float32'))
print(f"FAISS index created with {index.ntotal} vectors.")

# Step 5: Load the generative model
print("\n[Step 5] Loading generative model...")
print("Loading 'google/flan-t5-small' for text generation...")
tokenizer = T5Tokenizer.from_pretrained('google/flan-t5-small')
t5_model = T5ForConditionalGeneration.from_pretrained('google/flan-t5-small')
print("Model loaded successfully!")

# Step 6: Define the RAG pipeline
def answer_question(query):
    """RAG pipeline: Retrieve, Augment, Generate"""

    # RETRIEVE
    query_embedding = model.encode([query]).astype('float32')
    k = 2
    distances, indices = index.search(query_embedding, k)
    retrieved_chunks = [chunks[i] for i in indices[0]]
    context = "\n\n".join(retrieved_chunks)

    # AUGMENT
    prompt_template = f"""
Answer the following question using *only* the provided context.
If the answer is not in the context, say "I don't have that information."

Context:
{context}

Question:
{query}

Answer:
"""

    # GENERATE
    input_ids = tokenizer(prompt_template, return_tensors="pt", max_length=512, truncation=True).input_ids
    outputs = t5_model.generate(input_ids, max_length=100)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"--- CONTEXT ---\n{context}\n")
    return answer

# Step 7: Test the RAG system
print("\n" + "=" * 60)
print("Testing the RAG System")
print("=" * 60)

# Test 1: Question about WFH policy
print("\n[Test 1] Question about WFH policy:")
query_1 = "What is the WFH policy?"
print(f"Query: {query_1}")
answer_1 = answer_question(query_1)
print(f"Answer: {answer_1}\n")

# Test 2: Question not in the knowledge base
print("\n[Test 2] Question NOT in the knowledge base:")
query_2 = "What is the company's dental plan?"
print(f"Query: {query_2}")
answer_2 = answer_question(query_2)
print(f"Answer: {answer_2}\n")

# Test 3: Question about tech stack
print("\n[Test 3] Question about tech stack:")
query_3 = "What technologies does the company use for development?"
print(f"Query: {query_3}")
answer_3 = answer_question(query_3)
print(f"Answer: {answer_3}\n")

# Test 4: Question about PTO
print("\n[Test 4] Question about PTO:")
query_4 = "How many PTO days do employees get?"
print(f"Query: {query_4}")
answer_4 = answer_question(query_4)
print(f"Answer: {answer_4}\n")

print("=" * 60)
print("RAG System Demo Complete!")
print("=" * 60)
