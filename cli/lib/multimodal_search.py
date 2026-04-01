from PIL import Image
from sentence_transformers import SentenceTransformer
from lib.semantic_search import *
import json

class MultimodalSearch:
    def __init__(self, documents, model_name="clip-ViT-B-32"):
        self.model = SentenceTransformer(model_name)
        self.documents = documents
        self.texts = []
        for doc in documents:
            self.texts.append(f"{doc['title']}: {doc['description']}")
        self.text_embeddings = self.model.encode(self.texts, show_progress_bar=True)

    def embed_image(self, image_path):
        img = Image.open(image_path)
        return self.model.encode(img)
    
    def search_with_image(self, image_path):
        embedding = self.embed_image(image_path)
        i = 0
        results = []
        for te in self.text_embeddings:
            score = cosine_similarity(te, embedding)
            results.append({"score": score, "document": self.documents[i]})
            i += 1
        
        results = sorted(results, key=lambda x:x["score"], reverse=True)

        return results[:5]

    
def verify_image_embedding(image_path):
    search = MultimodalSearch()
    embedding = search.embed_image(image_path)
    print(f"Embedding shape: {embedding.shape[0]} dimensions")

def image_search_command(image_path):
    with open("data/movies.json", "r") as f:
        movies = json.load(f)
        
    documents = movies["movies"]    
    search = MultimodalSearch(documents)

    return search.search_with_image(image_path)
