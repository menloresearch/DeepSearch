#!/bin/bash
# Script to run model comparison between base model and LoRA model

# Initialize variables
LORA_PATH=""
TEMPERATURE=0.5

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --lora_path)
      LORA_PATH="$2"
      shift 2
      ;;
    --temperature)
      TEMPERATURE="$2"
      shift 2
      ;;
    --output_file)
      echo "Warning: Custom output_file is not recommended. Files are automatically saved in checkpoint's eval_logs directory."
      # We'll silently ignore this parameter
      shift 2
      ;;
    *)
      echo "Unknown option: $1"
      echo "Usage: $0 [--lora_path <path_to_checkpoint>] [--temperature <value>]"
      exit 1
      ;;
  esac
done

# If LORA_PATH is not provided, try to find the latest checkpoint
if [ -z "$LORA_PATH" ]; then
    echo "No checkpoint path provided, searching for latest checkpoint..."
    # Look for trainer_output directories in current directory and convert to absolute path
    TRAINER_DIR=$(find . -maxdepth 1 -type d -name "trainer_output_*" | sort -r | head -n 1)
    
    if [ -z "$TRAINER_DIR" ]; then
        echo "Error: No trainer output directory found. Please provide a checkpoint path with --lora_path"
        echo "Usage: $0 [--lora_path <path_to_checkpoint>] [--temperature <value>]"
        exit 1
    fi
    
    # Convert to absolute path
    TRAINER_DIR=$(realpath "$TRAINER_DIR")
    echo "Found trainer directory: ${TRAINER_DIR}"
    
    # Get the checkpoint path, filtering out log messages but keeping the path
    LORA_PATH=$(python -c "from eval import find_latest_checkpoint; print(find_latest_checkpoint('${TRAINER_DIR}') or '')" | grep -v "INFO" | grep -v "DEBUG" | grep -v "WARNING" | grep -v "ERROR" | grep -v "LangChain" | grep -v "FAISS" | grep -v "Successfully" | grep -v "Loading" | grep -v "Project root" | grep -v "Running in" | grep -v "Automatically" | grep -v "Platform" | grep -v "Torch" | grep -v "CUDA" | grep -v "Triton" | grep -v "Bfloat16" | grep -v "Free license" | grep -v "Fast downloading" | grep -v "vLLM loading" | grep -v "==" | grep -v "^$" | tail -n 1)
    
    if [ -z "$LORA_PATH" ]; then
        echo "Error: No checkpoint found in ${TRAINER_DIR}. Please provide a checkpoint path with --lora_path"
        echo "Usage: $0 [--lora_path <path_to_checkpoint>] [--temperature <value>]"
        exit 1
    fi
    echo "Found latest checkpoint: ${LORA_PATH}"
else
    # If LORA_PATH is provided, convert it to absolute path
    LORA_PATH=$(realpath "$LORA_PATH")
    # Get the trainer directory (parent of checkpoint directory)
    TRAINER_DIR=$(dirname "$(dirname "$LORA_PATH")")
fi

# Verify checkpoint and trainer directory exist
if [ ! -d "$(dirname "$LORA_PATH")" ]; then
    echo "Error: Checkpoint directory does not exist: $(dirname "$LORA_PATH")"
    exit 1
fi

if [ ! -d "$TRAINER_DIR" ]; then
    echo "Error: Trainer directory does not exist: $TRAINER_DIR"
    exit 1
fi

# Create eval_logs directory in the trainer output directory
EVAL_LOGS_DIR="$TRAINER_DIR/eval_logs"
mkdir -p "$EVAL_LOGS_DIR"

echo "Starting model comparison..."
echo "LoRA path: ${LORA_PATH}"
echo "Trainer directory: ${TRAINER_DIR}"
echo "Temperature: ${TEMPERATURE}"
echo "Evaluation logs will be saved in: ${EVAL_LOGS_DIR}"

# Run the comparison script, explicitly passing the trainer directory
python eval.py \
  --lora_path "${LORA_PATH}" \
  --temperature "${TEMPERATURE}" \
  --trainer_dir "${TRAINER_DIR}"

echo "Model comparison completed."
echo "Evaluation logs are saved in: ${EVAL_LOGS_DIR}" 