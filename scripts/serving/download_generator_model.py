import argparse
import os

from dotenv import load_dotenv
from huggingface_hub import snapshot_download

from config import GENERATOR_MODEL_DIR, GENERATOR_MODEL_REPO_ID


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(description="Download model from HuggingFace Hub")
    parser.add_argument(
        "--repo-id",
        type=str,
        default=GENERATOR_MODEL_REPO_ID,
        help="HuggingFace repository ID",
    )
    parser.add_argument(
        "--local-dir",
        type=str,
        default=GENERATOR_MODEL_DIR,
        help="Local directory to save model",
    )

    return parser.parse_args()


def main():
    """Main function to download model."""
    args = parse_args()
    load_dotenv(override=True)

    # Configuration
    HF_TOKEN = os.getenv("HF_TOKEN")

    print("Downloading model to", args.local_dir)

    # Download the model
    snapshot_download(
        token=HF_TOKEN,
        repo_id=args.repo_id,
        local_dir=args.local_dir,
        repo_type="model",
    )
    print(f"✅ Done: {args.repo_id} -> {args.local_dir}")


if __name__ == "__main__":
    main()
