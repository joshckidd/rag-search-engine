import json
import string

class MovieSearch:

    def __init__(self):
        with open("data/movies.json", "r") as f:
            self.movie_data = json.load(f)

    def title_search(self, query):
        results = []
        titles = []
        query_tokens = query.lower().translate(str.maketrans("", "", string.punctuation)).split()
        query_list = list(filter(None, query_tokens))
        for movie in self.movie_data["movies"]:
            title_tokens = movie["title"].lower().translate(str.maketrans("", "", string.punctuation)).split()
            title_list = list(filter(None, title_tokens))

            for q in query_list:
                for t in title_list:
                    if q in t and movie["title"] not in titles:
                        results.append(movie)
                        titles.append(movie["title"])
        return results
    
        
