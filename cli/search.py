import string
from nltk.stem import PorterStemmer
from inverted_index import InvertedIndex

class MovieSearch:

    def __init__(self):
        self.index = InvertedIndex()

    def search(self, query):
        results = []
        titles = []
        stemmer = PorterStemmer()
        query_tokens = query.lower().translate(str.maketrans("", "", string.punctuation)).split()
        query_list = list(filter(None, query_tokens))

        for q in query_list:
            if q not in self.index.stopwords:
                qtoken = stemmer.stem(q)
                docs = self.index.get_documents(qtoken)
                for doc in docs:
                    movie = self.index.docmap[doc]
                    if movie["title"] not in titles:
                        results.append(movie)
                        titles.append(movie["title"])
                    if len(results) == 5:
                        return results
        return results
    
        
