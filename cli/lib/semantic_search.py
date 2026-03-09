from sentence_transformers import SentenceTransformer
import numpy as np
import os
import json

class SemanticSearch:

    def __init__(self):
        # Load the model (downloads automatically the first time)
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.embeddings = None
        self.documents = None
        self.document_map = {}

    def generate_embedding(self, text):
        if text.strip() == "":
            raise ValueError("Text must have something in it.")
        
        embedding = self.model.encode([text])

        return embedding[0]
    
    def build_embeddings(self, documents):
        self.documents = documents
        doc_list = []
        for doc in self.documents:
            self.document_map[doc["id"]] = doc
            doc_list.append(f"{doc['title']}: {doc['description']}")

        self.embeddings = self.model.encode(doc_list, show_progress_bar=True)
        with open("cache/movie_embeddings.npy", "wb") as f:
            np.save(f, self.embeddings)

        return self.embeddings
    
    def load_or_create_embeddings(self, documents):
        self.documents = documents
        doc_list = []
        for doc in self.documents:
            self.document_map[doc["id"]] = doc
            doc_list.append(f"{doc['title']}: {doc['description']}")
        
        if os.path.exists("cache/movie_embeddings.npy"):
            with open("cache/movie_embeddings.npy", "rb") as f:
                self.embeddings = np.load(f)
                if len(self.embeddings) == len(documents):
                    return self.embeddings
                
        return self.build_embeddings(documents)
    
    def search(self, query, limit):
        if len(self.embeddings) == 0:
            raise ValueError("No embeddings loaded. Call `load_or_create_embeddings` first.")
        
        query_embedding = self.generate_embedding(query)

        scores = []

        for i in range(len(self.embeddings)):
            scores.append((cosine_similarity(query_embedding, self.embeddings[i]), i + 1))
        
        scores = sorted(scores, key=lambda x:x[0], reverse=True)

        res = []

        for i in range(limit):
            res.append({"score" : scores[i][0], "title": self.document_map[scores[i][1]]["title"], "description": self.document_map[scores[i][1]]["description"]})
            
        return res

def verify_model():
    model = SemanticSearch()

    print(f"Model loaded: {model.model}")
    print(f"Max sequence length: {model.model.max_seq_length}")

def embed_text(text):
    model = SemanticSearch()
    embedding = model.generate_embedding(text)
    print(f"Text: {text}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")

def verify_embeddings():
    model = SemanticSearch()
    with open("data/movies.json", "r") as f:
        movies = json.load(f)
    documents = movies["movies"]
    embeddings = model.load_or_create_embeddings(documents)

    print(f"Number of docs:   {len(documents)}")
    print(f"Embeddings shape: {embeddings.shape[0]} vectors in {embeddings.shape[1]} dimensions")

def embed_query_text(query):
    model = SemanticSearch()
    embedding = model.generate_embedding(query)

    print(f"Query: {query}")
    print(f"First 5 dimensions: {embedding[:5]}")
    print(f"Shape: {embedding.shape}")

def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)