import argparse
import json
from lib.hybrid_search import *

def main():
    parser = argparse.ArgumentParser(description="Search Evaluation CLI")
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Number of results to evaluate (k for precision@k, recall@k)",
    )

    args = parser.parse_args()
    limit = args.limit

    # run evaluation logic here
    with open("data/golden_dataset.json", "r") as f:
        dataset = json.load(f)

    full_results = []

    with open("data/movies.json", "r") as f:
        movies = json.load(f)
        documents = movies["movies"]

    model = HybridSearch(documents)

    for test_case in dataset["test_cases"]:

        results = model.rrf_search(test_case["query"], 60, args.limit)
        total_retrieved = len(results)
        total_relevant = len(test_case["relevant_docs"])
        relevant_retrieved = 0
        retrieved = ""
        relevant = ""
        for result in results:
            retrieved += f"{result["document"]["title"]}, "
            if result["document"]["title"] in test_case["relevant_docs"]:
                relevant_retrieved += 1
                relevant += f"{result["document"]["title"]}, "
        precision = relevant_retrieved / total_retrieved
        recall = relevant_retrieved / total_relevant
        retrieved = retrieved[:-2]
        relevant = relevant[:-2]
        f1 = 2 * (precision * recall) / (precision + recall)

        full_results.append({
            "precision": precision,
            "retrieved": retrieved,
            "relevant": relevant,
            "query": test_case["query"],
            "recall": recall,
            "f1": f1
        })

    print(f"k={args.limit}")
    print("")

    for fr in full_results:
        print(f"- Query: {fr["query"]}")
        print(f"  - Precision@{args.limit}: {fr["precision"]:.4f}")
        print(f"  - Recall@{args.limit}: {fr["recall"]:.4f}")
        print(f"  - F1 Score: {fr["f1"]:.4f}")
        print(f"  - Retrieved: {fr["retrieved"]}")
        print(f"  - Relevant: {fr["relevant"]}")
        print("")

if __name__ == "__main__":
    main()