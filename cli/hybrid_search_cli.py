import argparse
import json
from lib.hybrid_search import *
import os
import time
import json
from dotenv import load_dotenv
from google import genai
from sentence_transformers import CrossEncoder

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
    rrf_search_parser.add_argument("--enhance", type=str, choices=["spell", "rewrite", "expand"], help="Query enhancement method")
    rrf_search_parser.add_argument("--rerank-method", type=str, choices=["individual", "batch", "cross_encoder"], help="Results rerank method")
    rrf_search_parser.add_argument("--evaluate", action="store_true", help="Evaluate results flag")

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
            query = args.query

            load_dotenv()
            api_key = os.environ.get("GEMINI_API_KEY")
            if not api_key:
                raise RuntimeError("GEMINI_API_KEY environment variable not set")

            client = genai.Client(api_key=api_key)

            match args.enhance:
                case "spell":

                    content = f"""Fix any spelling errors in the user-provided movie search query below.
    Correct only clear, high-confidence typos. Do not rewrite, add, remove, or reorder words.
    Preserve punctuation and capitalization unless a change is required for a typo fix.
    If there are no spelling errors, or if you're unsure, output the original query unchanged.
    Output only the final query text, nothing else.
    User query: "{query}"
"""
                    response = client.models.generate_content(model="gemma-3-27b-it", contents=content)
                    query = response.text
                    print(f"Enhanced query ({args.enhance}): '{args.query}' -> '{query}'\n")
                case "rewrite":

                    content = f"""Rewrite the user-provided movie search query below to be more specific and searchable.

Consider:
- Common movie knowledge (famous actors, popular films)
- Genre conventions (horror = scary, animation = cartoon)
- Keep the rewritten query concise (under 10 words)
- It should be a Google-style search query, specific enough to yield relevant results
- Don't use boolean logic

Examples:
- "that bear movie where leo gets attacked" -> "The Revenant Leonardo DiCaprio bear attack"
- "movie about bear in london with marmalade" -> "Paddington London marmalade"
- "scary movie with bear from few years ago" -> "bear horror movie 2015-2020"

If you cannot improve the query, output the original unchanged.
Output only the rewritten query text, nothing else.

User query: "{query}"
"""
                    response = client.models.generate_content(model="gemma-3-27b-it", contents=content)
                    query = response.text
                    print(f"Enhanced query ({args.enhance}): '{args.query}' -> '{query}'\n")
                case "expand":

                    content = f"""Expand the user-provided movie search query below with related terms.

Add synonyms and related concepts that might appear in movie descriptions.
Keep expansions relevant and focused.
Output only the additional terms; they will be appended to the original query.

Examples:
- "scary bear movie" -> "scary horror grizzly bear movie terrifying film"
- "action movie with bear" -> "action thriller bear chase fight adventure"
- "comedy with bear" -> "comedy funny bear humor lighthearted"

User query: "{query}"
"""
                    response = client.models.generate_content(model="gemma-3-27b-it", contents=content)
                    query = response.text
                    print(f"Enhanced query ({args.enhance}): '{args.query}' -> '{query}'\n")

            limit = args.limit

            if args.rerank_method != None:
                limit = limit * 5

            results = model.rrf_search(query, args.k, limit)

            match args.rerank_method:
                case "individual":
                    for result in results:
                        doc = result["document"]
                        content = f"""Rate how well this movie matches the search query.

Query: "{query}"
Movie: {doc.get("title", "")} - {doc.get("description", "")}

Consider:
- Direct relevance to query
- User intent (what they're looking for)
- Content appropriateness

Rate 0-10 (10 = perfect match).
Output ONLY the number in your response, no other text or explanation.

Score:"""
                        score_response = client.models.generate_content(model="gemma-3-27b-it", contents=content)
                        result["rerank_score"] = score_response.text
                        time.sleep(3)
                    new_results = sorted(results, key=lambda x: x["rerank_score"], reverse=True)
                case "batch":
                    doc_list_str = str(results)
                    content = f"""Rank the movies listed below by relevance to the following search query.

Query: "{query}"

Movies:
{doc_list_str}

Return ONLY the movie IDs in order of relevance (best match first). Return a valid JSON list, nothing else.

For example:
[75, 12, 34, 2, 1]

Ranking:"""
                    score_response = client.models.generate_content(model="gemma-3-27b-it", contents=content)
                    top_results = json.loads(score_response.text)
                    new_results = []
                    i = 0
                    while i < args.limit and i < len(top_results):
                        for r in results:
                            if r["document"]["id"] == top_results[i]:
                                new_results.append(r)
                        i += 1
                case "cross_encoder":
                    pairs = []
                    for result in results:
                        doc = result["document"]
                        pairs.append([query, f"{doc.get('title', '')} - {doc.get('document', '')}"])
                    cross_encoder = CrossEncoder("cross-encoder/ms-marco-TinyBERT-L2-v2")
                    scores = cross_encoder.predict(pairs)
                    for i in range(len(results)):
                        results[i]["cross_encoder_score"] = scores[i]
                    new_results = sorted(results, key=lambda x: x["cross_encoder_score"], reverse=True)
                case _:
                    new_results = results
            print("Results before reranking:")
            for i in range(len(results)):
                print(f"{i + 1}. {results[i]["document"]["title"]}")
                match args.rerank_method:
                    case "batch":
                        print(f"   Re-rank Rank: {i + 1}")
                    case "individual":
                        print(f"   Re-rank Score: {results[i]["rerank_score"]:.3f}/10")
                    case "cross_encoder":
                        print(f"   Re-rank Score: {results[i]["cross_encoder_score"]:.3f}/10")
                print(f"   Hybrid Score: {results[i]["hybrid_score"]:.3f}")
                print(f"   BM25: {results[i]["keyword_score"]}, Semantic: {results[i]["semantic_score"]}")
                print(f"   {results[i]["document"]["description"][:80]}...")

            print("Results after reranking:")
            formatted_results = ""
            for i in range(args.limit):
                formatted_results += f"{i + 1}. {new_results[i]["document"]["title"]}\n"
                match args.rerank_method:
                    case "batch":
                       formatted_results += f"   Re-rank Rank: {i + 1}\n"
                    case "individual":
                        formatted_results += f"   Re-rank Score: {new_results[i]["rerank_score"]:.3f}/10\n"
                    case "cross_encoder":
                        formatted_results += f"   Re-rank Score: {new_results[i]["cross_encoder_score"]:.3f}/10\n"
                formatted_results += f"   Hybrid Score: {new_results[i]["hybrid_score"]:.3f}\n"
                formatted_results += f"   BM25: {new_results[i]["keyword_score"]}, Semantic: {new_results[i]["semantic_score"]}\n"
                formatted_results += f"   {new_results[i]["document"]["description"][:80]}...\n"
            print(formatted_results)

            if args.evaluate:
                content = f"""Rate how relevant each result is to this query on a 0-3 scale:

Query: "{query}"

Results:
{chr(10).join(formatted_results)}

Scale:
- 3: Highly relevant
- 2: Relevant
- 1: Marginally relevant
- 0: Not relevant

Do NOT give any numbers other than 0, 1, 2, or 3.

Return ONLY the scores in the same order you were given the documents. Return a valid JSON list, nothing else. For example:

[2, 0, 3, 2, 0, 1]"""
                response = client.models.generate_content(model="gemma-3-27b-it", contents=content)
                evaluation = json.loads(response.text)
                for i in range(args.limit):
                    print(f"{i + 1}. {new_results[i]["document"]["title"]}: {evaluation[i]}/3")
                
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()