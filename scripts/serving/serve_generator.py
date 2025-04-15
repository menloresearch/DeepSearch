import subprocess
import sys
from pathlib import Path

# Add project root to sys.path to allow importing config
# Assuming the script is at DeepSearch/scripts/serving/serve_generator.py
# The project root (DeepSearch) is parents[2]
PROJ_ROOT = Path(__file__).resolve().parents[2]
if str(PROJ_ROOT) not in sys.path:
    sys.path.append(str(PROJ_ROOT))

# Import after adjusting sys.path
try:
    from config import (
        GENERATOR_MODEL_REPO_ID,
        GENERATOR_SERVER_PORT,
        MODEL_CONFIG,
        logger,
    )
except ImportError as e:
    # Use print here as logger might not be available if import failed
    print(
        f"Error importing config: {e}. Make sure config.py is in the project root ({PROJ_ROOT}) and added to sys.path."
    )
    sys.exit(1)


def launch_sglang_server(
    model_id: str,
    port: int,
    context_length: int,
    host: str = "0.0.0.0",
    dtype: str = "bfloat16",
) -> None:
    """Launches the SGLang server using specified configurations.

    Args:
        model_id: The Hugging Face repository ID of the model.
        port: The port number for the server.
        context_length: The maximum context length for the model.
        host: The host address for the server.
        dtype: The data type for the model (e.g., 'bfloat16', 'float16').
    """
    command = [
        sys.executable,  # Use the current Python interpreter
        "-m",
        "sglang.launch_server",
        "--model-path",
        model_id,
        "--context-length",
        str(context_length),
        "--enable-metrics",
        "--dtype",
        dtype,
        "--host",
        host,
        "--port",
        str(port),
        "--mem-fraction-static",
        "0.5",
        "--trust-remote-code",
        # Recommended by SGLang for stability sometimes
        "--disable-overlap",
        # Can sometimes cause issues
        "--disable-radix-cache",
    ]

    # Log the command clearly
    command_str = " ".join(command)
    logger.info(f"🚀 Launching SGLang server with command: {command_str}")

    process = None  # Initialize process to None
    try:
        # Use Popen to start the server process
        # It runs in the foreground relative to this script,
        # but allows us to catch KeyboardInterrupt cleanly.
        process = subprocess.Popen(command)
        # Wait for the process to complete (e.g., user interruption)
        process.wait()
        # Check return code after waiting
        if process.returncode != 0:
            logger.error(f"💥 SGLang server process exited with error code: {process.returncode}")
            sys.exit(process.returncode)
        else:
            logger.info("✅ SGLang server process finished gracefully.")

    except FileNotFoundError:
        logger.error(f"💥 Error: Python executable or sglang module not found.")
        logger.error(f"Ensure '{sys.executable}' is correct and sglang is installed.")
        sys.exit(1)
    except KeyboardInterrupt:
        logger.info("🛑 SGLang server launch interrupted by user. Stopping server...")
        # Attempt to terminate the process gracefully
        if process and process.poll() is None:  # Check if process exists and is running
            process.terminate()
            try:
                process.wait(timeout=5)  # Wait a bit for termination
                logger.info("✅ Server terminated gracefully.")
            except subprocess.TimeoutExpired:
                logger.warning("⚠️ Server did not terminate gracefully, forcing kill.")
                process.kill()
        sys.exit(0)  # Exit cleanly after interrupt
    except Exception as e:
        # Catch any other unexpected exceptions during launch or waiting
        logger.error(f"🚨 An unexpected error occurred: {e}")
        # Ensure process is cleaned up if it exists
        if process and process.poll() is None:
            process.kill()
        sys.exit(1)


if __name__ == "__main__":
    # Get context length from config, default to 8192
    context_len = MODEL_CONFIG.get("max_seq_length", 8192)

    logger.info("----------------------------------------------------")
    logger.info("✨ Starting SGLang Generator Server ✨")
    logger.info(f"   Model ID: {GENERATOR_MODEL_REPO_ID}")
    logger.info(f"   Port: {GENERATOR_SERVER_PORT}")
    logger.info(f"   Context Length: {context_len}")
    logger.info("----------------------------------------------------")

    launch_sglang_server(
        model_id=GENERATOR_MODEL_REPO_ID,
        port=GENERATOR_SERVER_PORT,
        context_length=context_len,
    )
