import string
import json
import pickle
from nltk.stem import PorterStemmer
from pathlib import Path
from collections import Counter

class InvertedIndex:
    
    def __init__(self):
        self.index = {}
        self.docmap = {}
        self.term_frequencies = {}
        self.stemmer = PorterStemmer()
        with open("data/stopwords.txt", "r") as f:
            self.stopwords = f.read().splitlines()
        with open("data/movies.json", "r") as f:
            self.movie_data = json.load(f)

    def __add_document(self, doc_id, text):
        tokens = text.lower().translate(str.maketrans("", "", string.punctuation)).split()
        tokens = list(filter(None, tokens))

        token_list = []

        for token in tokens:
            if token not in self.stopwords:
                stem_token = self.stemmer.stem(token)
                token_list.append(stem_token)

                if stem_token in self.index:
                    doc_list = self.index[stem_token]
                else:
                    doc_list = []
                if doc_id not in doc_list:
                    doc_list.append(doc_id)
                self.index[stem_token] = doc_list
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
        tokens = term.lower().translate(str.maketrans("", "", string.punctuation)).split()
        if len(tokens) > 1:
            raise Exception("Too many terms.")
        stem_token = self.stemmer.stem(tokens[0])
        tf = self.term_frequencies[doc_id]
        if stem_token in tf:
            return tf[stem_token]
        return 0