from nltk.stem import PorterStemmer
import string

def tokenize(text):
    stemmer = PorterStemmer()
    with open("data/stopwords.txt", "r") as f:
        stopwords = f.read().splitlines()

    tokens = text.lower().translate(str.maketrans("", "", string.punctuation)).split()
    tokens = list(filter(None, tokens))

    token_list = []

    for token in tokens:
        if token not in stopwords:
            stem_token = stemmer.stem(token)
            token_list.append(stem_token)

    return token_list
