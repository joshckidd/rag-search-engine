import os

from .keyword_search import InvertedIndex
from .semantic_search import ChunkedSemanticSearch


class HybridSearch:
    def __init__(self, documents):
        self.documents = documents
        self.semantic_search = ChunkedSemanticSearch()
        self.semantic_search.load_or_create_chunk_embeddings(documents)

        self.idx = InvertedIndex()
        if not os.path.exists(self.idx.index_path):
            self.idx.build()
            self.idx.save()

    def _bm25_search(self, query, limit):
        self.idx.load()
        return self.idx.bm25_search(query, limit)

    def weighted_search(self, query, alpha, limit=5):
        keyword_results = self._bm25_search(query, limit*500)
        semantic_results = self.semantic_search.search_chunks(query, limit*500)

        keyword_scores = []
        for kr in keyword_results:
            keyword_scores.append(kr["score"])

        normalized_keyword_scores = normalize_scores(keyword_scores)

        semantic_scores = []
        for sr in semantic_results:
            semantic_scores.append(sr["score"])

        normalized_semantic_scores = normalize_scores(semantic_scores)

        results_dict = {}

        for i in range(len(keyword_results)):
            results_dict[keyword_results[i]["id"]] = {
                "document": keyword_results[i]["doc"],
                "keyword_score": normalized_keyword_scores[i],
                "semantic_score": 0.0
            }

        for i in range(len(semantic_results)):
            if semantic_results[i]["id"] in results_dict:
                results_dict[semantic_results[i]["id"]]["semantic_score"] = normalized_semantic_scores[i]
            else:
                results_dict[semantic_results[i]["id"]] = {
                    "document": semantic_results[i]["doc"],
                    "semantic_score": normalized_semantic_scores[i],
                    "keyword_score": 0.0
                }                

        for id in results_dict:
            results_dict[id]["hybrid_score"] = alpha * results_dict[id]["keyword_score"] + (1 - alpha) * results_dict[id]["semantic_score"]

        sorted_results = {k: v for k, v in sorted(results_dict.items(), key=lambda item: item[1]["hybrid_score"], reverse=True)}

        limited_results = []
        keys = list(sorted_results.keys())
        for i in range(limit):
            limited_results.append(sorted_results[keys[i]])

        return limited_results

        
    def rrf_search(self, query, k, limit=10):
        keyword_results = self._bm25_search(query, limit*500)
        semantic_results = self.semantic_search.search_chunks(query, limit*500)

        keyword_scores = []
        for kr in keyword_results:
            keyword_scores.append(kr["score"])

        semantic_scores = []
        for sr in semantic_results:
            semantic_scores.append(sr["score"])

        results_dict = {}

        for i in range(len(keyword_results)):
            results_dict[keyword_results[i]["id"]] = {
                "document": keyword_results[i]["doc"],
                "keyword_score": 1 / (k + i + 1),
                "semantic_score": 0.0
            }

        for i in range(len(semantic_results)):
            if semantic_results[i]["id"] in results_dict:
                results_dict[semantic_results[i]["id"]]["semantic_score"] = 1 / (k + i + 1)
            else:
                results_dict[semantic_results[i]["id"]] = {
                    "document": semantic_results[i]["doc"],
                    "semantic_score": 1 / (k + i + 1),
                    "keyword_score": 0.0
                }                

        for id in results_dict:
            results_dict[id]["hybrid_score"] = results_dict[id]["keyword_score"] + results_dict[id]["semantic_score"]

        sorted_results = {k: v for k, v in sorted(results_dict.items(), key=lambda item: item[1]["hybrid_score"], reverse=True)}

        limited_results = []
        keys = list(sorted_results.keys())
        for i in range(limit):
            limited_results.append(sorted_results[keys[i]])

        return limited_results

def normalize_scores(scores):
    if len(scores) == 0:
        return

    min = scores[0]
    max = scores[0]

    for score in scores:
        if min > score:
            min = score
        if max < score:
            max = score

    res = []
    for score in scores:
        if min == max :
            res.append(1.0)
        else:
            res.append((score - min) / (max - min))
        
    return res
