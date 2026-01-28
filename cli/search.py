import string
from tokenize import tokenize
from inverted_index import InvertedIndex

class MovieSearch:

    def __init__(self):
        self.index = InvertedIndex()

    def search(self, query):
        results = []
        titles = []
        query_list = tokenize(query)

        for qtoken in query_list:
            docs = self.index.get_documents(qtoken)
            for doc in docs:
                movie = self.index.docmap[doc]
                if movie["title"] not in titles:
                    results.append(movie)
                    titles.append(movie["title"])
                if len(results) == 5:
                    return results
        return results
    
        
