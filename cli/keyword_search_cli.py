#!/usr/bin/env python3

import argparse
import sys
from search import MovieSearch
from inverted_index import InvertedIndex

def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    search_parser = subparsers.add_parser("build", help="Build an inverted index of movies")

    search_parser = subparsers.add_parser("tf", help="Get the frequency of a term")
    search_parser.add_argument("doc_id", type=int, help="Document id")
    search_parser.add_argument("term", type=str, help="Term")

    search_parser = subparsers.add_parser("idf", help="Get the inverse document frequency of a term")
    search_parser.add_argument("term", type=str, help="Term")

    search_parser = subparsers.add_parser("tfidf", help="Get the frequency of a term")
    search_parser.add_argument("doc_id", type=int, help="Document id")
    search_parser.add_argument("term", type=str, help="Term")

    args = parser.parse_args()

    movies = MovieSearch()

    match args.command:
        case "search":
            try:
                movies.index.load()
            except Exception as e:
                print(e)
                sys.exit()
            print(f"Searching for: {args.query}")
            results = movies.search(args.query)
            for i in range(len(results)):
                print(f"{i+1}. {results[i]["title"]}")
        case "build":
            movies.index.build()
            movies.index.save()
        case "tf":
            try: 
                movies.index.load()
                tf = movies.index.get_tf(args.doc_id, args.term)
            except Exception as e:
                print(e)
                sys.exit()
            print(tf)
        case "idf":
            try: 
                movies.index.load()
                idf = movies.index.get_idf(args.term)
            except Exception as e:
                print(e)
                sys.exit()
            print(f"Inverse document frequency of '{args.term}': {idf:.2f}")
        case "tfidf":
            try: 
                movies.index.load()
                tf_idf = movies.index.get_tf_idf(args.doc_id, args.term)
            except Exception as e:
                print(e)
                sys.exit()
            print(f"TF-IDF score of '{args.term}' in document '{args.doc_id}': {tf_idf:.2f}")
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()