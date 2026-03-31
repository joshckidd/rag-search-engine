import argparse
from lib.multimodal_search import *

def main() -> None:
    parser = argparse.ArgumentParser(description="Multimodal Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    normalize_parser = subparsers.add_parser("verify_image_embedding", help="Verify an image embedding")
    normalize_parser.add_argument("image_path", type=str, help="Image path")

    args = parser.parse_args()

    match args.command:
        case "verify_image_embedding":
            verify_image_embedding(args.image_path)

if __name__ == "__main__":
    main()