import os
import sys
from datetime import datetime
from pathlib import Path

import torch
from dotenv import load_dotenv
from loguru import logger
from vllm import SamplingParams

# Load environment variables from .env file if it exists
load_dotenv(override=True)

# Project paths
PROJ_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJ_ROOT / "data"
MODEL_DIR = PROJ_ROOT / "models"
LOG_FOLDER = PROJ_ROOT / "logs"

# Evaluations
RETRIEVER_MODEL_REPO_ID = "intfloat/e5-base-v2"
RETRIEVER_MODEL_DIR = MODEL_DIR / "retriever"
RETRIEVER_SERVER_PORT = 8001
GENERATOR_MODEL_REPO_ID = "janhq/250404-llama-3.2-3b-instruct-grpo-03-s250"
GENERATOR_MODEL_DIR = MODEL_DIR / "generator"
GENERATOR_SERVER_PORT = 8002


# Model configuration
# MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
# MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"
MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"
# MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
# MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
# MODEL_NAME = "unsloth/Qwen2-1.5B"  # Smoke test first
device_id = 1 if os.environ.get("CUDA_VISIBLE_DEVICES") == "1" else torch.cuda.current_device()
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

OUTPUT_DIR = PROJ_ROOT / f"trainer_output_{MODEL_NAME.replace('/', '_')}_gpu{device_id}_{timestamp}"

# Model parameters
MODEL_CONFIG = {
    "max_seq_length": 4096 * 6,  # 24k tokens -> just try to utiliiz
    "lora_rank": 64,  # Larger rank = smarter, but slower
    "gpu_memory_utilization": 0.6,  # Reduce if out of memory
    "model_name": MODEL_NAME,
    "target_modules": [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],  # Remove QKVO if out of memory
}

# Training parameters
TRAINING_CONFIG = {
    "learning_rate": 5e-6,
    "adam_beta1": 0.9,
    "adam_beta2": 0.99,
    "weight_decay": 0.1,
    "warmup_ratio": 0.1,
    "lr_scheduler_type": "cosine",
    "optim": "paged_adamw_8bit",
    "logging_steps": 1,
    "per_device_train_batch_size": 8,
    "gradient_accumulation_steps": 1,  # Increase to 4 for smoother training
    "num_generations": 6,  # Decrease if out of memory
    "max_prompt_length": 4096 * 4 - 2048,
    "max_completion_length": 2048,
    "max_steps": 1000,
    "save_steps": 50,
    "max_grad_norm": 0.1,
    "report_to": "tensorboard",
}


# Sampling parameters
def get_sampling_params(temperature: float = 0.1) -> SamplingParams:
    """Get sampling parameters for text generation"""
    return SamplingParams(
        temperature=temperature,
        top_p=0.95,
        max_tokens=4096 * 6,
    )


# Initialize logging based on environment
def _init_logging(env: str = "development") -> None:
    """
    Initialize logging configuration with console logging
    and default file logging to ./logs directory.
    Additional file logging will be set up later in update_log_path().

    Args:
        env: The environment for logging ('development' or 'production')
    """
    # Create default log folder
    if not LOG_FOLDER.exists():
        LOG_FOLDER.mkdir(parents=True, exist_ok=True)

    # Remove any existing handlers
    logger.remove()

    # Define the logging format
    console_format = (
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> "
        "| <level>{level: <8}</level> "
        "| <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> "
        "- <level>{message}</level>"
    )

    file_format = "{time:YYYY-MM-DD at HH:mm:ss} | {level} | {name}:{function}:{line} - {message}"

    # Add console logging with INFO level (minimal terminal output)
    logger.add(
        sys.stderr,
        format=console_format,
        level="INFO",  # "INFO",  # Changed from DEBUG to INFO for minimal terminal output
        colorize=True,
        backtrace=True,
        diagnose=True,
    )

    # Add default file logging to ./logs directory with DEBUG level (full details)
    logger.add(
        LOG_FOLDER / "app.log",
        format=file_format,
        level="DEBUG",  # Keep DEBUG level for full file logging
        rotation="500 MB",
        retention="7 days",
        compression="zip",
        enqueue=True,  # Enables asynchronous logging
    )

    # Add custom level for requests
    logger.level("REQUEST", no=25, color="<yellow>", icon=" ")

    # Configure exception handling
    def exception_handler(exc_type, exc_value, exc_traceback):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return
        logger.opt(exception=(exc_type, exc_value, exc_traceback)).critical("Unhandled exception")

    sys.excepthook = exception_handler


# Update the log files to point to the training directory
def update_log_path(log_dir=None):
    """
    Add a log file in the training directory while keeping the default ./logs logging.
    Should be called after the training directory is created.

    Args:
        log_dir: Path to store additional log files (default: uses get_paths()["log_dir"])
    """
    # Use provided log_dir or get from training paths
    if log_dir is None:
        paths = get_paths(create_dirs=True)
        log_dir = paths["log_dir"]
    else:
        log_dir = Path(log_dir)
        log_dir.mkdir(exist_ok=True, parents=True)

    file_format = "{time:YYYY-MM-DD at HH:mm:ss} | {level} | {name}:{function}:{line} - {message}"

    # Add additional file handler pointing to training directory
    # No need to remove existing handlers as we want to keep those
    logger.add(
        log_dir / "app.log",
        format=file_format,
        level="INFO",
        rotation="500 MB",
        retention="7 days",
        compression="zip",
        enqueue=True,  # Enables asynchronous logging
    )

    logger.info(f"Additional logs will be stored in: {log_dir}")


# Paths configuration without creating directories
def get_paths(create_dirs: bool = False) -> dict:
    """
    Get common paths for the project

    Args:
        create_dirs: Whether to create the directories

    Returns:
        Dictionary with paths
    """
    output_dir = Path(OUTPUT_DIR)
    log_dir = output_dir / "logs"
    tensorboard_dir = output_dir / "runs"

    # Only create directories if explicitly requested
    if create_dirs:
        output_dir.mkdir(exist_ok=True)
        log_dir.mkdir(exist_ok=True)

        # Only create tensorboard directory if it's enabled in config
        if TRAINING_CONFIG.get("report_to") == "tensorboard":
            tensorboard_dir.mkdir(exist_ok=True)

    return {
        "output_dir": output_dir,
        "log_dir": log_dir,
        "tensorboard_dir": tensorboard_dir,
        "proj_root": PROJ_ROOT,
        "data_dir": DATA_DIR,
    }


# Create training directories
def init_training_dirs():
    """Initialize all directories needed for training"""
    paths = get_paths(create_dirs=True)

    # Also ensure our standard project directories exist
    for directory in [
        DATA_DIR,
        LOG_FOLDER,
    ]:
        directory.mkdir(exist_ok=True, parents=True)

    return paths


# For backward compatibility - will be deprecated
def setup_logger(module_name=None, create_dirs: bool = False):
    """
    Setup a logger for a specific module with consistent configuration.

    Note: This function is kept for backward compatibility.
    Use the global 'logger' instead for new code.

    Args:
        module_name: Optional name of module for module-specific log file
        create_dirs: Whether to create log directories

    Returns:
        Configured logger instance
    """
    logger.warning("setup_logger is deprecated. Import logger directly from config instead.")
    return logger


# Initialize logging on module import
env = os.getenv("APP_ENV", "development")
_init_logging(env=env)

# Log project root on import
logger.info(f"Project root path: {PROJ_ROOT}")
logger.debug(f"Running in {env} environment")


if __name__ == "__main__":
    print(PROJ_ROOT)
