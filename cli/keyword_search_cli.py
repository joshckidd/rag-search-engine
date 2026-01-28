#!/usr/bin/env python3

import argparse
import sys
from search import MovieSearch
from settings import *

def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    build_parser = subparsers.add_parser("build", help="Build an inverted index of movies")

    tf_parser = subparsers.add_parser("tf", help="Get the frequency of a term")
    tf_parser.add_argument("doc_id", type=int, help="Document id")
    tf_parser.add_argument("term", type=str, help="Term")

    idf_parser = subparsers.add_parser("idf", help="Get the inverse document frequency of a term")
    idf_parser.add_argument("term", type=str, help="Term")

    tf_idf_parser = subparsers.add_parser("tfidf", help="Get the tf-idf of a term")
    tf_idf_parser.add_argument("doc_id", type=int, help="Document id")
    tf_idf_parser.add_argument("term", type=str, help="Term")

    bm25_idf_parser = subparsers.add_parser("bm25idf", help="Get the bm25-idf of a term")
    bm25_idf_parser.add_argument("term", type=str, help="Term")

    bm25_tf_parser = subparsers.add_parser(
    "bm25tf", help="Get BM25 TF score for a given document ID and term"
    )
    bm25_tf_parser.add_argument("doc_id", type=int, help="Document ID")
    bm25_tf_parser.add_argument("term", type=str, help="Term to get BM25 TF score for")
    bm25_tf_parser.add_argument("k1", type=float, nargs='?', default=BM25_K1, help="Tunable BM25 K1 parameter")
    bm25_tf_parser.add_argument("b", type=float, nargs='?', default=BM25_B, help="Tunable BM25 b parameter")

    bm25search_parser = subparsers.add_parser("bm25search", help="Search movies using full BM25 scoring")
    bm25search_parser.add_argument("query", type=str, help="Search query")

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
        case "bm25idf":
            try: 
                movies.index.load()
                bm25idf = movies.index.get_bm25_idf(args.term)
            except Exception as e:
                print(e)
                sys.exit()
            print(f"BM25 IDF score of '{args.term}': {bm25idf:.2f}")
        case "bm25tf":
            try: 
                movies.index.load()
                bm25tf = movies.index.get_bm25_tf(args.doc_id, args.term, args.k1, args.b)
            except Exception as e:
                print(e)
                sys.exit()
            print(f"BM25 TF score of '{args.term}' in document '{args.doc_id}': {bm25tf:.2f}")
        case "bm25search":
            try: 
                movies.index.load()
                results = movies.index.bm25_search(args.query, 5)
            except Exception as e:
                print(e)
                sys.exit()
            for i in range(len(results)):
                print(f"{i+1}. ({results[i]["id"]}) {results[i]["doc"]["title"]} - Score: {results[i]["score"]:.2f}")
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()