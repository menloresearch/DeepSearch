"""Upload local directory to HuggingFace Hub.
This script uploads a specified local directory to HuggingFace Hub as a private repository.

Example:
    python upload_checkpoint.py --local-dir "models/my_model" --repo-id "org/model-name"
"""

import argparse
import os

from dotenv import load_dotenv
from huggingface_hub import HfApi


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(description="Upload model to HuggingFace Hub")
    parser.add_argument("--local-dir", type=str, required=True, help="Local directory to upload")
    parser.add_argument("--repo-id", type=str, required=True, help="HuggingFace repository ID")
    parser.add_argument("--public", action="store_true", help="Make repository public (default: private)")
    return parser.parse_args()


def main():
    """Main function to upload model."""
    args = parse_args()
    load_dotenv(override=True)

    # Configuration
    HF_TOKEN = os.getenv("HF_TOKEN")

    # Files to ignore during upload
    IGNORE_PATTERNS = [
        "*.log",  # Log files
        "*.pyc",  # Python cache
        ".git*",  # Git files
        "*.bin",  # Binary files
        "*.pt",  # PyTorch checkpoints
        "*.ckpt",  # Checkpoints
        "events.*",  # Tensorboard
        "wandb/*",  # Weights & Biases
        "runs/*",  # Training runs
    ]

    api = HfApi(token=HF_TOKEN)
    api.create_repo(repo_id=args.repo_id, private=not args.public, exist_ok=True, repo_type="model")
    api.upload_folder(
        folder_path=args.local_dir,
        repo_id=args.repo_id,
        repo_type="model",
        #  ignore_patterns=IGNORE_PATTERNS
    )
    print(f"✅ Done: {args.local_dir} -> {args.repo_id}")


if __name__ == "__main__":
    main()
