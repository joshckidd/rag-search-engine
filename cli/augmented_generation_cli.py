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
    summarize_parser = subparsers.add_parser(
        "summarize", help="Summarize results"
    )
    summarize_parser.add_argument("query", type=str, help="Search query for RAG")
    summarize_parser.add_argument("--limit", type=int, default=5, help="Maximum number of results to return")
    citations_parser = subparsers.add_parser(
        "citations", help="Summarize results with citations"
    )
    citations_parser.add_argument("query", type=str, help="Search query for RAG")
    citations_parser.add_argument("--limit", type=int, default=5, help="Maximum number of results to return")
    question_parser = subparsers.add_parser(
        "question", help="Answer a question with search results"
    )
    question_parser.add_argument("question", type=str, help="The question to ask")
    question_parser.add_argument("--limit", type=int, default=5, help="Maximum number of results to return")

    args = parser.parse_args()

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

    match args.command:
        case "rag":
            query = args.query
            # do RAG stuff here
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
        case "summarize":
            query = args.query
            limit = args.limit

            # do RAG stuff here
            results = model.rrf_search(query, 60, limit)

            prompt = f"""Provide information useful to the query below by synthesizing data from multiple search results in detail.

The goal is to provide comprehensive information so that users know what their options are.
Your response should be information-dense and concise, with several key pieces of information about the genre, plot, etc. of each movie.

This should be tailored to Hoopla users. Hoopla is a movie streaming service.

Query: {query}

Search results:
{results}

Provide a comprehensive 3–4 sentence answer that combines information from multiple sources:"""
            response = client.models.generate_content(model="gemma-3-27b-it", contents=prompt)

            print("Search Results:")
            for doc in results:
                print(f"- {doc["document"]["title"]}")
            print("")
            print("LLM Summary:")
            print(response.text)            
        case "citations":
            query = args.query
            limit = args.limit

            # do RAG stuff here
            documents = model.rrf_search(query, 60, limit)

            prompt = prompt = f"""Answer the query below and give information based on the provided documents.

The answer should be tailored to users of Hoopla, a movie streaming service.
If not enough information is available to provide a good answer, say so, but give the best answer possible while citing the sources available.

Query: {query}

Documents:
{documents}

Instructions:
- Provide a comprehensive answer that addresses the query
- Cite sources in the format [1], [2], etc. when referencing information
- If sources disagree, mention the different viewpoints
- If the answer isn't in the provided documents, say "I don't have enough information"
- Be direct and informative

Answer:"""
            response = client.models.generate_content(model="gemma-3-27b-it", contents=prompt)

            print("Search Results:")
            for doc in documents:
                print(f"- {doc["document"]["title"]}")
            print("")
            print("LLM Summary:")
            print(response.text)            
        case "question":
            question = args.question
            limit = args.limit

            # do RAG stuff here
            context = model.rrf_search(question, 60, limit)

            prompt = prompt = prompt = f"""Answer the user's question based on the provided movies that are available on Hoopla, a streaming service.

Question: {question}

Documents:
{context}

Instructions:
- Answer questions directly and concisely
- Be casual and conversational
- Don't be cringe or hype-y
- Talk like a normal person would in a chat conversation

Answer:"""
            response = client.models.generate_content(model="gemma-3-27b-it", contents=prompt)

            print("Search Results:")
            for doc in context:
                print(f"- {doc["document"]["title"]}")
            print("")
            print("Answer:")
            print(response.text)            
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()