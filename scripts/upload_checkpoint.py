"""Upload local directory to HuggingFace Hub.
This script uploads a specified local directory to HuggingFace Hub as a private repository.
It uses API token from HuggingFace for authentication.
"""

import os

from dotenv import load_dotenv
from huggingface_hub import HfApi

load_dotenv(override=True)

# Configuration
LOCAL_DIR = "trainer_output_deepseek-ai_DeepSeek-R1-Distill-Qwen-7B_gpu0_20250403_050520"
REPO_ID = "janhq/250403-runpod-qwen7b-r1-distil"
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
api.create_repo(repo_id=REPO_ID, private=True, exist_ok=True, repo_type="model")
api.upload_folder(
    folder_path=LOCAL_DIR,
    repo_id=REPO_ID,
    repo_type="model",
    # ignore_patterns=IGNORE_PATTERNS,
)
print(f"✅ Done: {LOCAL_DIR} -> {REPO_ID}")
