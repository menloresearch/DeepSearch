import argparse
import os
import zipfile

from dotenv import load_dotenv
from huggingface_hub import snapshot_download

from config import DATA_DIR


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(description="Download FlashRAG datasets from HuggingFace Hub")
    parser.add_argument(
        "--repo-id",
        type=str,
        default="RUC-NLPIR/FlashRAG_datasets",
        help="HuggingFace repository IDs",
    )
    parser.add_argument(
        "--local-dir",
        type=str,
        default=DATA_DIR / "flashrag_datasets",
        help="Local directory to save model",
    )

    return parser.parse_args()


def main():
    """Main function to download model."""
    args = parse_args()
    load_dotenv(override=True)

    # Configuration
    HF_TOKEN = os.getenv("HF_TOKEN")

    ALLOW_PATTERNS = [
        "*retrieval-corpus*",
        "*bamboogle*",
        "*nq*",
    ]

    # Download the model
    snapshot_download(
        token=HF_TOKEN,
        repo_id=args.repo_id,
        local_dir=args.local_dir,
        repo_type="dataset",
        # ignore_patterns=IGNORE_PATTERNS,
        allow_patterns=ALLOW_PATTERNS,
    )

    # unzip data/flashrag_datasets/retrieval-corpus/wiki18_100w.zip
    print("Unzipping wiki18_100w.zip. Might take a while...")
    zip_file_path = os.path.join(args.local_dir, "retrieval-corpus", "wiki18_100w.zip")
    with zipfile.ZipFile(zip_file_path, "r") as zip_ref:
        zip_ref.extractall(args.local_dir)

    print(f"✅ Done: {args.repo_id} -> {args.local_dir}")


if __name__ == "__main__":
    main()
