"""Download model from HuggingFace Hub.
This script downloads a model repository from HuggingFace Hub to local directory.
"""

import os

from dotenv import load_dotenv
from huggingface_hub import snapshot_download

load_dotenv(override=True)

# Configuration
REPO_ID = "janhq/250403-runpod-qwen7b-r1-distil"
LOCAL_DIR = "downloaded_model"  # Where to save the model
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
    repo_id=REPO_ID,
    local_dir=LOCAL_DIR,
    # ignore_patterns=IGNORE_PATTERNS,
)
print(f"✅ Done: {REPO_ID} -> {LOCAL_DIR}")
