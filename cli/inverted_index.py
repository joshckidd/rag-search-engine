import string
import json
import pickle
import math
from pathlib import Path
from collections import Counter
from tokenize import tokenize

class InvertedIndex:
    
    def __init__(self):
        self.index = {}
        self.docmap = {}
        self.term_frequencies = {}
        with open("data/movies.json", "r") as f:
            self.movie_data = json.load(f)

    def __add_document(self, doc_id, text):
        token_list = tokenize(text)

        for token in token_list:
            if token in self.index:
                doc_list = self.index[token]
            else:
                doc_list = []
            if doc_id not in doc_list:
                doc_list.append(doc_id)
            self.index[token] = doc_list
        self.term_frequencies[doc_id] = Counter(token_list)

    def get_documents(self, term):
        if term.lower() in self.index:
            doc_list = self.index[term.lower()]
        else:
            doc_list = []
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
        with open("cache/term_frequencies.pkl", "wb") as f:
            pickle.dump(self.term_frequencies, f)

    def load(self):
        try:
            with open("cache/index.pkl", "rb") as f:
                self.index = pickle.load(f)
            with open("cache/docmap.pkl", "rb") as f:
                self.docmap = pickle.load(f)
            with open("cache/term_frequencies.pkl", "rb") as f:
                self.term_frequencies = pickle.load(f)
        except:
            raise Exception("File not found.")
        
    def get_tf(self, doc_id, term):
        tokens = tokenize(term)
        if len(tokens) > 1:
            raise Exception("Too many terms.")
        tf = self.term_frequencies[doc_id]
        if tokens[0] in tf:
            return tf[tokens[0]]
        return 0
    
    def get_idf(self, term):
        total_doc_count = len(self.docmap)
        term_match_doc_count = 0

        for doc in self.docmap:
            if self.get_tf(int(doc), term) > 0:
                term_match_doc_count += 1
        
        return math.log((total_doc_count + 1) / (term_match_doc_count + 1))
    
    def get_tf_idf(self, doc_id, term):
        return self.get_tf(doc_id, term) * self.get_idf(term)
