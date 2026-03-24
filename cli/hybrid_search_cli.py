import argparse
import json
from lib.hybrid_search import *

def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    normalize_parser = subparsers.add_parser("normalize", help="Normalize a list of scores")
    normalize_parser.add_argument("scores", type=float, nargs="+", help="Search query")

    weighted_search_parser = subparsers.add_parser("weighted-search", help="Perform a weighted hybrid search")
    weighted_search_parser.add_argument("query", type=str, help="Search query")
    weighted_search_parser.add_argument("--alpha", type=float, default=0.5, help="Alpha value to determine weights")
    weighted_search_parser.add_argument("--limit", type=int, default=5, help="Maximum number of results to return")

    rrf_search_parser = subparsers.add_parser("rrf-search", help="Perform an rrf hybrid search")
    rrf_search_parser.add_argument("query", type=str, help="Search query")
    rrf_search_parser.add_argument("-k", type=int, default=60, help="K value to rank weights")
    rrf_search_parser.add_argument("--limit", type=int, default=5, help="Maximum number of results to return")

    args = parser.parse_args()

    match args.command:
        case "normalize":
            scores = normalize_scores(args.scores)
            for score in scores:
                print(f"* {score:.4f}")
        case "weighted-search":
            with open("data/movies.json", "r") as f:
                movies = json.load(f)
                documents = movies["movies"]

            model = HybridSearch(documents)
                
            results = model.weighted_search(args.query, args.alpha, args.limit)
            
            for i in range(len(results)):
                print(f"{i + 1}. {results[i]["document"]["title"]}")
                print(f"   Hybrid Score: {results[i]["hybrid_score"]:.3f}")
                print(f"   BM25: {results[i]["keyword_score"]:.3f}, Semantic: {results[i]["semantic_score"]:.3f}")
                print(f"   {results[i]["document"]["description"][:80]}...")
        case "rrf-search":
            with open("data/movies.json", "r") as f:
                movies = json.load(f)
                documents = movies["movies"]

            model = HybridSearch(documents)
                
            results = model.rrf_search(args.query, args.k, args.limit)
            
            for i in range(len(results)):
                print(f"{i + 1}. {results[i]["document"]["title"]}")
                print(f"   Hybrid Score: {results[i]["hybrid_score"]:.3f}")
                print(f"   BM25: {results[i]["keyword_score"]}, Semantic: {results[i]["semantic_score"]}")
                print(f"   {results[i]["document"]["description"][:80]}...")
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()