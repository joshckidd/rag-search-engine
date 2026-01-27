#!/usr/bin/env python3

import argparse
from search import MovieSearch
from inverted_index import InvertedIndex

def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    search_parser = subparsers.add_parser("build", help="Build an inverted index of movies")

    args = parser.parse_args()

    movies = MovieSearch()
    index = InvertedIndex()

    match args.command:
        case "search":
            print(f"Searching for: {args.query}")
            results = movies.title_search(args.query)
            for i in range(len(results)):
                print(f"{i+1}. {results[i]["title"]}")
        case "build":
            index.build()
            index.save()
            docs = index.get_documents("merida")
            print(f"First document for token 'merida' = {docs[0]}")
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()