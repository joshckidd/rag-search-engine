from sentence_transformers import SentenceTransformer
import numpy as np
import os
import json
import re

class SemanticSearch:

    def __init__(self, model_name='all-MiniLM-L6-v2'):
        # Load the model (downloads automatically the first time)
        self.model = SentenceTransformer(model_name)
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
        for doc in self.documents:
            self.document_map[doc["id"]] = doc
        
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
    
class ChunkedSemanticSearch(SemanticSearch):
    def __init__(self, model_name = "all-MiniLM-L6-v2") -> None:
        super().__init__(model_name)
        self.chunk_embeddings = None
        self.chunk_metadata = None

    def build_chunk_embeddings(self, documents):
        self.documents = documents
        chunk_list = []
        chunk_metadata = []
        chunk_index = 0

        for i in range(len(self.documents)):
            doc = self.documents[i]
            self.document_map[doc["id"]] = doc
            if doc["description"] != "":
                chunks = semantic_chunk(doc["description"], 4, 1)
                for c in chunks:
                    chunk_list.append(c)
                    chunk_metadata.append({"movie_idx": i + 1, "chunk_idx": chunk_index, "total_chunks": len(chunks)})
                    chunk_index += 1

        self.chunk_embeddings = self.model.encode(chunk_list, show_progress_bar=True)
        self.chunk_metadata = {"chunks": chunk_metadata, "total_chunks": len(chunk_list)}
        with open("cache/chunk_embeddings.npy", "wb") as f:
            np.save(f, self.chunk_embeddings)

        with open("cache/chunk_metadata.json", "w") as f:
            json.dump({"chunks": chunk_metadata, "total_chunks": len(chunk_list)}, f, indent=2)

        return self.chunk_embeddings

    def load_or_create_chunk_embeddings(self, documents: list[dict]) -> np.ndarray:
        self.documents = documents
        for doc in self.documents:
            self.document_map[doc["id"]] = doc
        
        if os.path.exists("cache/chunk_embeddings.npy") and os.path.exists("cache/chunk_metadata.json"):
            with open("cache/chunk_embeddings.npy", "rb") as f:
                self.chunk_embeddings = np.load(f)

            with open("cache/chunk_metadata.json", "r") as f:
                self.chunk_metadata = json.load(f)

            return self.chunk_embeddings
                
        return self.build_chunk_embeddings(documents)

    def search_chunks(self, query: str, limit: int = 10):
        
        query_embedding = self.generate_embedding(query)

        chunk_scores = []

        for i in range(len(self.chunk_embeddings)):
            cs = cosine_similarity(query_embedding, self.chunk_embeddings[i])
            chunk_scores.append({"chunk_idx": i,
                           "movie_idx": self.chunk_metadata["chunks"][i]["movie_idx"],
                           "score": cs})
            
        movie_scores = {}

        for s in chunk_scores:
            if s["movie_idx"] not in movie_scores or s["score"] > movie_scores[s["movie_idx"]]:
                movie_scores[s["movie_idx"]] = s["score"]
   
        movie_scores_ranked = []

        for ms in movie_scores:
            movie_scores_ranked.append({"movie_idx": ms, "score": movie_scores[ms]})

        movie_scores_ranked = sorted(movie_scores_ranked, key=lambda x:x["score"], reverse=True)

        res = []

        for i in range(limit):
            res.append({"score" : movie_scores_ranked[i]["score"], 
                        "title": self.document_map[movie_scores_ranked[i]["movie_idx"]]["title"], 
                        "document": self.document_map[movie_scores_ranked[i]["movie_idx"]]["description"][:100],
                        "id": movie_scores_ranked[i]["movie_idx"],
                        "metadata": {}})
            
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

def chunk(text, chunk_size, overlap):
    words = text.split()
    i = 0
    n = 1
    res = []
    while i  < len(words):
        res.append(" ".join(words[i:i + chunk_size]))
        i += chunk_size - overlap
        n += 1
    return res

def semantic_chunk(text, max_chunk_size, overlap):
    stripped_text = text.strip()
    if stripped_text == "":
        return []

    sentences = re.split(r"(?<=[.!?])\s+", stripped_text)
    if len(sentences) == 1 and sentences[0][:-1] not in ".!?":
        sentences = [stripped_text]
    
    stripped_sentences = []
    for sentence in sentences:
        stripped_sentence = sentence.strip()
        if stripped_sentence != "":
            stripped_sentences.append(sentence)

    i = 0
    n = 1
    res = []
    while i + overlap  < len(stripped_sentences):
        res.append(" ".join(stripped_sentences[i:i + max_chunk_size]))
        i += max_chunk_size - overlap
        n += 1
    return res