"""Download model from HuggingFace Hub.
This script downloads a model repository from HuggingFace Hub to local directory.

Example:
    python download_checkpoint.py --repo-id "org/model-name" --local-dir "models"
"""

import argparse
import os

from dotenv import load_dotenv
from huggingface_hub import snapshot_download


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(description="Download model from HuggingFace Hub")
    parser.add_argument(
        "--repo-id", type=str, default="janhq/250403-llama-3.2-3b-instruct-grpo", help="HuggingFace repository ID"
    )
    parser.add_argument("--local-dir", type=str, default="downloaded_model", help="Local directory to save model")

    return parser.parse_args()


def main():
    """Main function to download model."""
    args = parse_args()
    load_dotenv(override=True)

    # Configuration
    HF_TOKEN = os.getenv("HF_TOKEN")

    # Files to ignore during download
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

    # Download the model
    snapshot_download(
        token=HF_TOKEN,
        repo_id=args.repo_id,
        local_dir=args.local_dir,
        repo_type="model",
        # ignore_patterns=IGNORE_PATTERNS
    )
    print(f"✅ Done: {args.repo_id} -> {args.local_dir}")


if __name__ == "__main__":
    main()
