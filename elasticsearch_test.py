from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer
import time
import requests
# Connect to Elasticsearch
es = Elasticsearch("http://127.0.0.1:9200", http_auth=("elastic", "CYq3=005bpy3LzpZL93j")  # Basic authentication
)

try:
    # Check if Elasticsearch is reachable
    if es.ping():
        print("Successfully connected to Elasticsearch!")
    else:
        raise ValueError("Failed to ping Elasticsearch.")
except ValueError as ve:
    print(f"Connection error: {ve}")
except requests.exceptions.RequestException as e:
    print(f"Request exception: {e}")

# Define index mapping for dense_vector
index_name = "semantic_search"
if not es.indices.exists(index=index_name):
    es.indices.create(
        index=index_name,
        body={
            "mappings": {
                "properties": {
                    "text": {"type": "text"},
                    "embedding": {"type": "dense_vector", "dims": 384}
                }
            }
        }
    )

# Load sentence transformer model
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Documents to index
documents = [
    "This is the first document.",
    "This document is the second document."
]

# Generate embeddings
embeddings = model.encode(documents)

# Index documents with embeddings
for i, (doc, emb) in enumerate(zip(documents, embeddings)):
    es.index(index=index_name, id=i, body={"text": doc, "embedding": emb.tolist()})

# Wait a bit for indexing to complete
time.sleep(2)

# Semantic search query
query_text = "This is a query document."
query_embedding = model.encode([query_text])[0]

# KNN search (for Elasticsearch 8.x)
query = {
    "size": 2,
    "query": {
        "knn": {
            "field": "embedding",
            "query_vector": query_embedding.tolist(),
            "k": 2,
            "num_candidates": 10  # Optional for better results
        }
    }
}

# Execute search
res = es.search(index=index_name, body=query)

# Print results
print("Search results:")
for hit in res["hits"]["hits"]:
    print(hit["_source"]["text"])
