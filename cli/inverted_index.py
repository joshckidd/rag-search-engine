import string
import json
import pickle
from nltk.stem import PorterStemmer
from pathlib import Path

class InvertedIndex:
    
    def __init__(self):
        self.index = {}
        self.docmap = {}
        with open("data/stopwords.txt", "r") as f:
            self.stopwords = f.read().splitlines()
        with open("data/movies.json", "r") as f:
            self.movie_data = json.load(f)

    def __add_document(self, doc_id, text):
        stemmer = PorterStemmer()

        tokens = text.lower().translate(str.maketrans("", "", string.punctuation)).split()
        tokens = query_list = list(filter(None, tokens))

        for token in tokens:
            if token not in self.stopwords:
                stem_token = stemmer.stem(token)

                if token in self.index:
                    doc_list = self.index[token]
                else:
                    doc_list = []
                if doc_id not in doc_list:
                    doc_list.append(doc_id)
                self.index[token] = doc_list

    def get_documents(self, term):
        doc_list = self.index[term.lower()]
        return sorted(doc_list)
    
    def build(self):
        for movie in self.movie_data["movies"]:
            self.docmap[movie["id"]] = movie
            self.__add_document(movie["id"], f"{movie['title']} {movie['description']}")

    def save(self):
        Path("cache/").mkdir(parents=True, exist_ok=True)
        with open("cache/index.pkl", "wb") as f:
            pickle.dump(self.index, f)
        with open("cache/docmap.pkl", "wb") as f:
            pickle.dump(self.docmap, f)

