import json
import string
from nltk.stem import PorterStemmer

class MovieSearch:

    def __init__(self):
        with open("data/movies.json", "r") as f:
            self.movie_data = json.load(f)

        with open("data/stopwords.txt", "r") as f:
            self.stopwords = f.read().splitlines()

    def title_search(self, query):
        results = []
        titles = []
        stemmer = PorterStemmer()
        query_tokens = query.lower().translate(str.maketrans("", "", string.punctuation)).split()
        query_list = list(filter(None, query_tokens))
        for movie in self.movie_data["movies"]:
            title_tokens = movie["title"].lower().translate(str.maketrans("", "", string.punctuation)).split()
            title_list = list(filter(None, title_tokens))

            for q in query_list:
                if q not in self.stopwords:
                    qtoken = stemmer.stem(q)
                    for t in title_list:
                        if t not in self.stopwords:
                            ttoken = stemmer.stem(t)
                            if qtoken in ttoken and movie["title"] not in titles:
                                results.append(movie)
                                titles.append(movie["title"])
        return results
    
        
