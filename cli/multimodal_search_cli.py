import argparse
from lib.multimodal_search import *

def main() -> None:
    parser = argparse.ArgumentParser(description="Multimodal Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    verify_image_embedding_parser = subparsers.add_parser("verify_image_embedding", help="Verify an image embedding")
    verify_image_embedding_parser.add_argument("image_path", type=str, help="Image path")

    image_search_parser = subparsers.add_parser("image_search", help="Search by image")
    image_search_parser.add_argument("image_path", type=str, help="Image path")

    args = parser.parse_args()

    match args.command:
        case "verify_image_embedding":
            verify_image_embedding(args.image_path)
        case "image_search":
            results = image_search_command(args.image_path)
            i = 1
            for result in results:
                print(f"{i}. {result["document"]["title"]} (similarity: {result["score"]:.3f})")
                print(f"   {result["document"]["description"][:80]}...")
                print("")
                i += 1

if __name__ == "__main__":
    main()