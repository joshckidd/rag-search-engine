import os
import json
import pickle
import math
from pathlib import Path
from collections import Counter
from lib.tokenize import tokenize
from lib.settings import *

class InvertedIndex:
    
    def __init__(self):
        self.index = {}
        self.docmap = {}
        self.term_frequencies = {}
        self.doc_lengths = {}
        self.index_path = os.path.join(CACHE_DIR, "index.pkl")
        self.docmap_path = os.path.join(CACHE_DIR, "docmap.pkl")
        self.term_frequencies_path = os.path.join(CACHE_DIR, "term_frequencies.pkl")
        self.doc_lengths_path = os.path.join(CACHE_DIR, "doc_lengths.pkl")
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
        self.doc_lengths[doc_id] = len(token_list)

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
        Path(CACHE_DIR).mkdir(parents=True, exist_ok=True)
        with open(self.index_path, "wb") as f:
            pickle.dump(self.index, f)
        with open(self.docmap_path, "wb") as f:
            pickle.dump(self.docmap, f)
        with open(self.term_frequencies_path, "wb") as f:
            pickle.dump(self.term_frequencies, f)
        with open(self.doc_lengths_path, "wb") as f:
            pickle.dump(self.doc_lengths, f)

    def load(self):
        try:
            with open(self.index_path, "rb") as f:
                self.index = pickle.load(f)
            with open(self.docmap_path, "rb") as f:
                self.docmap = pickle.load(f)
            with open(self.term_frequencies_path, "rb") as f:
                self.term_frequencies = pickle.load(f)
            with open(self.doc_lengths_path, "rb") as f:
                self.doc_lengths = pickle.load(f)
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
        tokens = tokenize(term)
        if len(tokens) > 1:
            raise Exception("Too many terms.")
        term_match_doc_count = len(self.index[tokens[0]])
        
        return math.log((total_doc_count + 1) / (term_match_doc_count + 1))
    
    def get_tf_idf(self, doc_id, term):
        return self.get_tf(doc_id, term) * self.get_idf(term)
    
    def get_bm25_idf(self, term):
        total_doc_count = len(self.docmap)
        tokens = tokenize(term)
        if len(tokens) > 1:
            raise Exception("Too many terms.")
        if tokens[0] in self.index:
            term_match_doc_count = len(self.index[tokens[0]])
        else:
            term_match_doc_count = 0
        
        return math.log((total_doc_count - term_match_doc_count + 0.5) / (term_match_doc_count + 0.5) + 1)
    
    def get_bm25_tf(self, doc_id, term, k1=BM25_K1, b=BM25_B):
        tf = self.get_tf(doc_id, term)
        doc_length = self.doc_lengths[doc_id]
        avg_doc_length = self.__get_avg_doc_length()
        length_norm = 1 - b + b * (doc_length / avg_doc_length)
        return (tf * (k1 + 1)) / (tf + k1 * length_norm)
    
    def __get_avg_doc_length(self):
        if len(self.doc_lengths) == 0:
            return 0.0
        
        total_lengths = 0
        for doc in self.doc_lengths:
            total_lengths += self.doc_lengths[doc]

        return total_lengths / len(self.doc_lengths)
    
    def bm25(self, doc_id, term):
        return self.get_bm25_tf(doc_id, term) * self.get_bm25_idf(term)
    
    def bm25_search(self, query, limit):
        query_tokens = tokenize(query)
        bm25_scores = {}

        for doc in self.docmap:
            doc_terms_total = 0.0
            for term in query_tokens:
                doc_terms_total += self.bm25(doc, term)
            bm25_scores[doc] = doc_terms_total
        
        bm25_scores = {k: v for k, v in sorted(bm25_scores.items(), key=lambda item: item[1], reverse=True)}

        results = []

        for doc in bm25_scores:
            result = {}
            result["id"] = doc
            result["score"] = bm25_scores[doc]
            result["doc"] = self.docmap[doc]
            results.append(result)

            if len(results) == limit:
                return results
        
        return results


            
