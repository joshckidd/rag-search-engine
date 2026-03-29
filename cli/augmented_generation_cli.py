import argparse
import json
from lib.hybrid_search import *
from dotenv import load_dotenv
from google import genai

def main():
    parser = argparse.ArgumentParser(description="Retrieval Augmented Generation CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    rag_parser = subparsers.add_parser(
        "rag", help="Perform RAG (search + generate answer)"
    )
    rag_parser.add_argument("query", type=str, help="Search query for RAG")

    args = parser.parse_args()

    match args.command:
        case "rag":
            query = args.query
            # do RAG stuff here
            with open("data/golden_dataset.json", "r") as f:
                dataset = json.load(f)

            with open("data/movies.json", "r") as f:
                movies = json.load(f)
                documents = movies["movies"]

            load_dotenv()
            api_key = os.environ.get("GEMINI_API_KEY")
            if not api_key:
                raise RuntimeError("GEMINI_API_KEY environment variable not set")

            client = genai.Client(api_key=api_key)

            model = HybridSearch(documents)
            docs = model.rrf_search(query, 60, 5)

            prompt = f"""You are a RAG agent for Hoopla, a movie streaming service.
Your task is to provide a natural-language answer to the user's query based on documents retrieved during search.
Provide a comprehensive answer that addresses the user's query.

Query: {query}

Documents:
{docs}

Answer:"""
            response = client.models.generate_content(model="gemma-3-27b-it", contents=prompt)

            print("Search Results:")
            for doc in docs:
                print(f"- {doc["document"]["title"]}")
            print("")
            print("RAG Response:")
            print(response.text)
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()