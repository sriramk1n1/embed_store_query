import ollama
import chromadb

MODEL="nomic-embed-text"

def generate_embeddings(text):
    result = ollama.embeddings(model=MODEL, prompt=text)
    return result["embedding"]

def store(doc_id, text):
    embedding = generate_embeddings(text)
    collection.add(
        documents=[text],    
        ids=[doc_id],            
        embeddings=[embedding]    
    )
    print(f"Stored embedding for document {doc_id}")

def query(query_text, num_results=3):
    query_embedding = generate_embeddings(query_text)
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=num_results
    )
    return results

def split_text_into_chunks(text, chunk_size):
    chunks = []    
    for i in range(0, len(text), chunk_size):
        chunks.append(text[i:i + chunk_size])
    
    return chunks

client = chromadb.PersistentClient(path="./chroma_persist")
collection = client.get_or_create_collection(name="my_embeddings")

#Example

store("doc1", "Capital of India is New Delhi")
store("doc2", "Capital of China is Beijing")

query_text = "what is capital of India?"
results = query(query_text)

for i, result in enumerate(results['documents'][0]):
    print(f"Result {i+1}: {result}")
