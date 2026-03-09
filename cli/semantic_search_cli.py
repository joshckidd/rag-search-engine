#!/usr/bin/env python3

import argparse
from lib.semantic_search import *

def main():
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    verify_parser = subparsers.add_parser("verify", help="Verify semantic model")

    embed_text_parser = subparsers.add_parser("embed_text", help="Embed text")
    embed_text_parser.add_argument("text", type=str, help="Text")

    verify_embeddings_parser = subparsers.add_parser("verify_embeddings", help="Verify embeddings")

    embedquery_parser = subparsers.add_parser("embedquery", help="Embed query")
    embedquery_parser.add_argument("query", type=str, help="Query")

    search_parser = subparsers.add_parser("search", help="Seamantic search")
    search_parser.add_argument("query", type=str, help="Search query")
    search_parser.add_argument("--limit", type=int, default=5, help="Maximum results to return")

    chunk_parser = subparsers.add_parser("chunk", help="Chunk text")
    chunk_parser.add_argument("text", type=str, help="Text to chunk")
    chunk_parser.add_argument("--chunk-size", type=int, default=200, help="Number of words in a chunk")

    args = parser.parse_args()

    match args.command:
        case "verify":
            verify_model()
        case "embed_text":
            embed_text(args.text)
        case "verify_embeddings":
            verify_embeddings()
        case "embedquery":
            embed_query_text(args.query)
        case "search":
            model = SemanticSearch()
            with open("data/movies.json", "r") as f:
                movies = json.load(f)
                documents = movies["movies"]
                model.load_or_create_embeddings(documents)
                results = model.search(args.query, args.limit)

                for i in range(len(results)):
                    print(f"{i + 1}. {results[i]["title"]} ({results[i]["score"]})")
                    print(f"   {results[i]["description"]}")
        case "chunk":
            words = args.text.split()
            i = 0
            print(f"Chunking {len(args.text)} characters")
            while i * args.chunk_size < len(words):
                print(f"{i+1}. {" ".join(words[i * args.chunk_size:(i + 1) * args.chunk_size])}")
                i += 1
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()