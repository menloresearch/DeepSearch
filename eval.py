"""
Compare base model with LoRA model performance.

This script evaluates and compares the performance of a base model against
the same model with a LoRA adapter applied.
"""

import argparse
import glob
import os
import re
import time
from datetime import datetime

from unsloth import FastLanguageModel
from vllm import SamplingParams

import src.rl_helpers as rl_helpers
from src.config import MODEL_NAME, OUTPUT_DIR, logger


def find_latest_checkpoint(search_dir=None):
    """
    Find the latest checkpoint in the specified directory or OUTPUT_DIR.

    Args:
        search_dir: Directory to search for checkpoints (default: OUTPUT_DIR)

    Returns:
        Path to the latest checkpoint or None if no checkpoints found
    """
    if search_dir is None:
        search_dir = OUTPUT_DIR
        logger.info(f"No search directory provided, using default: {search_dir}")
    else:
        logger.info(f"Searching for checkpoints in: {search_dir}")

    # Check if the directory exists first
    if not os.path.exists(search_dir):
        logger.warning(f"Search directory {search_dir} does not exist")
        return None

    # First try to find checkpoints in the format checkpoint-{step}
    checkpoints = glob.glob(os.path.join(search_dir, "checkpoint-*"))

    if checkpoints:
        # Extract checkpoint numbers and sort
        checkpoint_numbers = []
        for checkpoint in checkpoints:
            match = re.search(r"checkpoint-(\d+)$", checkpoint)
            if match:
                checkpoint_numbers.append((int(match.group(1)), checkpoint))

        if checkpoint_numbers:
            # Sort by checkpoint number (descending)
            checkpoint_numbers.sort(reverse=True)
            latest = checkpoint_numbers[0][1]
            logger.info(f"Found latest checkpoint: {latest}")
            return latest

    # If no checkpoints found, look for saved_adapter_{timestamp}.bin files
    adapter_files = glob.glob(os.path.join(search_dir, "saved_adapter_*.bin"))
    if adapter_files:
        # Sort by modification time (newest first)
        adapter_files.sort(key=os.path.getmtime, reverse=True)
        latest = adapter_files[0]
        logger.info(f"Found latest adapter file: {latest}")
        return latest

    # If all else fails, look for any .bin files
    bin_files = glob.glob(os.path.join(search_dir, "*.bin"))
    if bin_files:
        # Sort by modification time (newest first)
        bin_files.sort(key=os.path.getmtime, reverse=True)
        latest = bin_files[0]
        logger.info(f"Found latest .bin file: {latest}")
        return latest

    logger.warning(f"No checkpoints found in {search_dir}")
    return None


def get_model_config():
    """Get model configuration."""
    return {
        "max_seq_length": 4096 * 2,
        "lora_rank": 64,
        "gpu_memory_utilization": 0.6,
        "model_name": MODEL_NAME,
        "target_modules": [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    }


def get_sampling_params(temperature: float = 0.5) -> SamplingParams:
    """Get sampling parameters for generation."""
    return SamplingParams(
        temperature=temperature,
        top_p=0.95,
        max_tokens=4096,
    )


def setup_model_and_tokenizer():
    """Initialize model and tokenizer with LoRA support."""
    config = get_model_config()
    logger.info(f"Setting up model {config['model_name']} with LoRA support...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config["model_name"],
        max_seq_length=config["max_seq_length"],
        load_in_4bit=True,
        fast_inference=True,
        max_lora_rank=config["lora_rank"],
        gpu_memory_utilization=config["gpu_memory_utilization"],
    )

    # Setup LoRA
    model = FastLanguageModel.get_peft_model(
        model,
        r=config["lora_rank"],
        target_modules=config["target_modules"],
        lora_alpha=config["lora_rank"],
        use_gradient_checkpointing=True,
        random_state=3407,
    )

    logger.info("Model and tokenizer setup complete.")
    return model, tokenizer


def test_lora_functionality(model, tokenizer, lora_path):
    """
    Test if LoRA is working properly by doing a direct comparison on a simple prompt.

    Args:
        model: The model to test
        tokenizer: The tokenizer
        lora_path: Path to LoRA weights

    Returns:
        bool: True if LoRA is working properly
    """
    logger.info(f"\n{'=' * 50}")
    logger.info("TESTING LORA FUNCTIONALITY")
    logger.info(f"{'=' * 50}")

    # First check if LoRA path exists
    if not os.path.exists(lora_path):
        logger.error(f"ERROR: LoRA path does not exist: {lora_path}")
        return False

    logger.info(f"LoRA path exists: {lora_path}")

    # Test prompt
    test_prompt = "Explain the concept of Low-Rank Adaptation (LoRA) in one paragraph:"

    # Format prompt for model
    formatted_prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": test_prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )

    # Sample with base model
    logger.info("Generating with base model...")
    sampling_params = get_sampling_params(temperature=0.7)  # Higher temp to make differences more obvious
    base_response = model.fast_generate(
        [formatted_prompt],
        sampling_params=sampling_params,
    )
    if hasattr(base_response[0], "outputs"):
        base_text = base_response[0].outputs[0].text
    else:
        base_text = base_response[0]

    # Sample with LoRA
    logger.info(f"Loading LoRA adapter from {lora_path}...")
    lora_request = model.load_lora(lora_path)
    if lora_request is None:
        logger.error("ERROR: Failed to load LoRA adapter")
        return False

    logger.info(f"LoRA adapter loaded successfully: {lora_request}")
    logger.info("Generating with LoRA model...")

    lora_response = model.fast_generate(
        [formatted_prompt],
        sampling_params=sampling_params,
        lora_request=lora_request,
    )
    if hasattr(lora_response[0], "outputs"):
        lora_text = lora_response[0].outputs[0].text
    else:
        lora_text = lora_response[0]

    # Check if responses are different
    are_identical = base_text == lora_text
    logger.info(f"\nResponses are {'identical' if are_identical else 'different'}")

    logger.info("\nBASE MODEL RESPONSE:")
    logger.info("-" * 40)
    logger.info(base_text[:500] + "..." if len(base_text) > 500 else base_text)
    logger.info("-" * 40)

    logger.info("\nLoRA MODEL RESPONSE:")
    logger.info("-" * 40)
    logger.info(lora_text[:500] + "..." if len(lora_text) > 500 else lora_text)
    logger.info("-" * 40)

    if are_identical:
        logger.warning("\nWARNING: LoRA adapter does not seem to change the model's output")
        logger.warning("This could indicate that the LoRA adapter is not being properly applied")
    else:
        logger.info("\nLoRA adapter is working as expected (outputs are different)")

    return not are_identical


def evaluate_model(
    model,
    tokenizer,
    lora_path=None,
    temperature=0.5,
    output_file="eval_results.txt",
    trainer_dir=None,
):
    """
    Evaluate model with or without LoRA weights.

    Args:
        model: The model to evaluate
        tokenizer: The tokenizer
        lora_path: Path to LoRA weights (None or empty for base model, "auto" for auto-detect)
        temperature: Sampling temperature
        output_file: File to write results to
        trainer_dir: Directory containing the checkpoints (parent of checkpoint directory)

    Returns:
        dict: Evaluation results
    """
    sampling_params = get_sampling_params(temperature=temperature)

    # --- Determine Trainer Output Directory ---
    # Prioritize the directory passed from the shell script if available
    if trainer_dir and os.path.isdir(trainer_dir):
        trainer_output_dir = os.path.abspath(trainer_dir)
        logger.info(f"Using trainer directory passed from arguments: {trainer_output_dir}")
    else:
        logger.warning(
            f"Trainer directory not provided or invalid: {trainer_dir}. Attempting to determine automatically."
        )
        # Fallback logic if trainer_dir is not provided or invalid
        temp_lora_path = lora_path
        if temp_lora_path == "auto":
            # Find latest checkpoint, searching within OUTPUT_DIR by default
            temp_lora_path = find_latest_checkpoint()  # Searches OUTPUT_DIR by default

        if temp_lora_path and os.path.exists(temp_lora_path):
            # If a LoRA path exists (provided or found), get its parent's parent
            checkpoint_dir = os.path.dirname(os.path.abspath(temp_lora_path))
            trainer_output_dir = os.path.dirname(checkpoint_dir)
            logger.info(f"Determined trainer directory from LoRA path ({temp_lora_path}): {trainer_output_dir}")
        else:
            # If no LoRA path, default to current directory (should ideally not happen if called from eval.sh)
            trainer_output_dir = os.path.abspath(".")
            logger.warning(
                f"Could not determine trainer directory automatically. Defaulting to current directory: {trainer_output_dir}"
            )

    # --- Auto-detect LoRA path if needed, searching within the determined trainer_output_dir ---
    if lora_path == "auto":
        # Pass the determined trainer_output_dir to find_latest_checkpoint
        detected_checkpoint = find_latest_checkpoint(search_dir=trainer_output_dir)
        if detected_checkpoint:
            lora_path = detected_checkpoint
            logger.info(f"Auto-detected latest checkpoint in {trainer_output_dir}: {lora_path}")
        else:
            logger.warning(f"No checkpoint found in {trainer_output_dir} for auto-detection. Evaluating base model.")
            lora_path = None

    model_type = "LoRA" if lora_path else "Base"

    logger.info(f"\n{'=' * 50}")
    logger.info(f"Starting evaluation of {model_type} model")
    logger.info(f"Trainer Output Directory: {trainer_output_dir}")  # Log the final directory
    logger.info(f"{'=' * 50}")

    # --- Create eval_logs directory ---
    # Always create it inside the determined trainer_output_dir
    eval_log_dir = os.path.join(trainer_output_dir, "eval_logs")
    try:
        os.makedirs(eval_log_dir, exist_ok=True)
        logger.info(f"Ensured eval_logs directory exists at: {eval_log_dir}")
    except OSError as e:
        logger.error(f"Failed to create directory {eval_log_dir}: {e}")
        # Fallback to current directory if creation fails
        eval_log_dir = os.path.abspath("./eval_logs")
        os.makedirs(eval_log_dir, exist_ok=True)
        logger.warning(f"Fell back to creating eval_logs in current directory: {eval_log_dir}")

    # Create file names based on model type
    model_prefix = "lora" if lora_path else "base"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Define all output file paths
    eval_log_file = os.path.join(eval_log_dir, f"{model_prefix}_model_eval_{timestamp}.log")
    output_file = os.path.join(eval_log_dir, f"{model_prefix}_model_results.txt")
    debug_file = os.path.join(eval_log_dir, f"{model_prefix}_model_results_debug.txt")

    logger.info(f"Writing evaluation log to: {eval_log_file}")
    logger.info(f"Results will be saved to: {output_file}")

    # Function to generate completions
    def eval_generate_fn(inputs):
        start_time = time.time()
        if lora_path:
            lora_request = model.load_lora(lora_path)
            load_time = time.time() - start_time
            logger.info(f"LoRA adapter loaded in {load_time:.2f} seconds: {lora_request}")
            responses = model.fast_generate(inputs, sampling_params=sampling_params, lora_request=lora_request)
        else:
            # For base model, add additional logging
            logger.info("Generating with base model (no LoRA)")
            # Also write to the base model log file directly
            with open(eval_log_file, "a") as f:
                f.write(f"\n{'=' * 50}\n")
                f.write("BASE MODEL GENERATION\n")
                f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Model: {MODEL_NAME}\n")
                f.write(f"Temperature: {temperature}\n")
                f.write(f"{'=' * 50}\n\n")

            responses = model.fast_generate(inputs, sampling_params=sampling_params)

        gen_time = time.time() - start_time
        logger.debug(f"Generation completed in {gen_time:.2f} seconds")
        return responses

    def verifier_generate_fn(inputs):
        # Use a lower temperature for verification to get more consistent results
        verifier_params = get_sampling_params(temperature=0.1)
        return model.fast_generate(inputs, sampling_params=verifier_params)

    # Prepare the verification function
    verify_fn = rl_helpers.build_reward_correctness_fn(verifier_generate_fn, tokenizer, log_file=eval_log_file)

    # Get the dataset and prepare questions and answers
    train_dataset, test_dataset = rl_helpers.get_qa_dataset()
    questions = test_dataset["prompt"]
    inputs = questions

    logger.info(f"Verifying {len(inputs)} answers...")

    # Run the evaluation
    start_time = time.time()
    logger.info(f"Starting {model_type} model evaluation...")

    full_chat_states = rl_helpers.run_eval(
        generate_fn=eval_generate_fn,
        verify_fn=verify_fn,
        tokenizer=tokenizer,
        output_file=output_file,
        debug_file=debug_file,
    )

    # Calculate rewards
    logger.info(f"Calculating rewards for {model_type} model...")
    rewards = verify_fn(questions, full_chat_states, answer=test_dataset["answer"])
    avg_reward = sum(rewards) / len(rewards)
    total_time = time.time() - start_time

    # Record the results
    results = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model_type": model_type,
        "model_name": MODEL_NAME,
        "lora_path": lora_path if lora_path else "None",
        "accuracy": avg_reward,
        "correct_count": sum(rewards),
        "total_count": len(rewards),
        "temperature": temperature,
        "time_taken": total_time,
    }

    # Add more detailed output to log file
    logger.info(f"\n{'=' * 50}")
    logger.info(f"{model_type.upper()} MODEL EVALUATION RESULTS:")
    logger.info(f"{'=' * 50}")
    logger.info(f"Accuracy: {avg_reward:.4f} ({sum(rewards)}/{len(rewards)} correct)")
    logger.info(f"Temperature: {temperature}")
    logger.info(f"Time taken: {total_time:.2f} seconds")
    logger.info(f"Results file: {output_file}")
    logger.info(f"Debug file: {debug_file}")
    logger.info(f"Log file: {eval_log_file}")

    # Write a summary to the log file too
    with open(eval_log_file, "a") as f:
        f.write(f"\n{'=' * 50}\n")
        f.write(f"{model_type.upper()} MODEL EVALUATION SUMMARY\n")
        f.write(f"{'=' * 50}\n")
        f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Accuracy: {avg_reward:.4f} ({sum(rewards)}/{len(rewards)} correct)\n")
        f.write(f"Temperature: {temperature}\n")
        f.write(f"Time taken: {total_time:.2f} seconds\n")
        f.write(f"Results saved to: {output_file}\n")
        f.write(f"Debug data saved to: {debug_file}\n\n")

    logger.info(f"Evaluation completed. Results saved to {output_file} and {debug_file}")

    return results


def compare_models(lora_path, temperature=0.5, output_file=None, trainer_dir=None):
    """
    Compare base model with LoRA model.

    Args:
        lora_path: Path to LoRA weights (use "auto" for auto-detection)
        temperature: Sampling temperature
        output_file: File to write results to (optional, will be auto-generated if None)
        trainer_dir: Directory containing the trainer output (parent of checkpoint directory)
    """
    # Auto-detect checkpoint if requested
    if lora_path == "auto":
        search_dir = trainer_dir if trainer_dir else OUTPUT_DIR
        detected_checkpoint = find_latest_checkpoint(search_dir=search_dir)
        if detected_checkpoint:
            lora_path = detected_checkpoint
            logger.info(f"Auto-detected latest checkpoint: {lora_path}")
        else:
            logger.warning("No checkpoint found for auto-detection. Skipping comparison.")
            return

    # Set up output directory in the checkpoint directory
    checkpoint_dir = os.path.dirname(lora_path)
    if not trainer_dir:
        trainer_dir = os.path.dirname(checkpoint_dir)

    eval_log_dir = os.path.join(trainer_dir, "eval_logs")
    os.makedirs(eval_log_dir, exist_ok=True)

    # Define the comparison file path if not provided
    if output_file is None:
        output_file = os.path.join(eval_log_dir, "model_comparison_results.txt")

    # Define file paths for individual model results
    base_output = os.path.join(eval_log_dir, "base_model_results.txt")
    lora_output = os.path.join(eval_log_dir, "lora_model_results.txt")

    model, tokenizer = setup_model_and_tokenizer()

    # Test if LoRA is working properly
    lora_works = test_lora_functionality(model, tokenizer, lora_path)
    if not lora_works:
        logger.warning("LoRA adapter test failed. Results may not be reliable.")

    # Evaluate both models
    base_results = evaluate_model(
        model,
        tokenizer,
        lora_path=None,
        temperature=temperature,
        output_file=base_output,
        trainer_dir=trainer_dir,
    )

    lora_results = evaluate_model(
        model,
        tokenizer,
        lora_path=lora_path,
        temperature=temperature,
        output_file=lora_output,
        trainer_dir=trainer_dir,
    )

    # Calculate improvement
    improvement = lora_results["accuracy"] - base_results["accuracy"]

    # Write comparison results
    with open(output_file, "w") as f:
        f.write("MODEL COMPARISON RESULTS\n")
        f.write("======================\n\n")
        f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Base Model: {MODEL_NAME}\n")
        f.write(f"LoRA Path: {lora_path}\n\n")
        f.write(f"Base Model Accuracy: {base_results['accuracy']:.4f}\n")
        f.write(f"LoRA Model Accuracy: {lora_results['accuracy']:.4f}\n")
        f.write(f"Improvement: {improvement:.4f}\n")
        f.write(f"Temperature: {temperature}\n")
        f.write(f"Base Model Time: {base_results['time_taken']:.2f}s\n")
        f.write(f"LoRA Model Time: {lora_results['time_taken']:.2f}s\n\n")
        f.write(f"Base Model Results File: {base_output}\n")
        f.write(f"LoRA Model Results File: {lora_output}\n")

    logger.info("\nModel comparison completed.")
    logger.info(f"\n{'=' * 50}")
    logger.info("MODEL COMPARISON RESULTS:")
    logger.info(f"{'=' * 50}")
    logger.info(f"Base Model Accuracy: {base_results['accuracy']:.4f}")
    logger.info(f"LoRA Model Accuracy: {lora_results['accuracy']:.4f}")
    logger.info(f"Improvement: {improvement:.4f}")
    logger.info(f"Temperature: {temperature}")
    logger.info(f"Results written to: {output_file}")
    logger.info(f"Base Model Results: {base_output}")
    logger.info(f"LoRA Model Results: {lora_output}")
    logger.info(f"{'=' * 50}")

    return {
        "base_accuracy": base_results["accuracy"],
        "lora_accuracy": lora_results["accuracy"],
        "improvement": improvement,
        "output_file": output_file,
        "base_output": base_output,
        "lora_output": lora_output,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate and compare models")
    parser.add_argument(
        "--lora_path",
        type=str,
        default="auto",
        help="Path to LoRA weights (use 'auto' for auto-detection)",
    )
    parser.add_argument("--temperature", type=float, default=0.5, help="Sampling temperature")
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="File to write results to (optional, will be auto-generated if None)",
    )
    parser.add_argument(
        "--trainer_dir",
        type=str,
        default=None,
        help="Directory containing the trainer output (parent of checkpoint directory)",
    )
    args = parser.parse_args()

    # Auto-detect checkpoint first to set up logging directory
    checkpoint_dir = None
    lora_path = args.lora_path
    trainer_dir = args.trainer_dir

    if trainer_dir:
        if os.path.exists(trainer_dir):
            logger.info(f"Using provided trainer directory: {trainer_dir}")
        else:
            logger.warning(f"Provided trainer directory does not exist: {trainer_dir}")
            trainer_dir = None

    if lora_path == "auto":
        search_dir = trainer_dir if trainer_dir else OUTPUT_DIR
        detected_checkpoint = find_latest_checkpoint(search_dir=search_dir)
        if detected_checkpoint:
            lora_path = detected_checkpoint
            checkpoint_dir = os.path.dirname(lora_path)
            if not trainer_dir:  # Only set if not provided
                trainer_dir = os.path.dirname(checkpoint_dir)

            # Set up logging in the trainer directory
            eval_log_dir = os.path.join(trainer_dir, "eval_logs")
            os.makedirs(eval_log_dir, exist_ok=True)

            # If this is imported from config, use it here
            try:
                from src.config import update_log_path

                update_log_path(eval_log_dir)
                logger.info(f"Logs will be saved to both ./logs and {eval_log_dir}")
            except ImportError:
                logger.info("Config's update_log_path not available, using default logging")

    if trainer_dir:
        logger.info(f"Using trainer directory: {trainer_dir}")
        logger.info(f"All evaluation files will be stored in: {os.path.join(trainer_dir, 'eval_logs')}")
    else:
        logger.warning("No trainer directory found, will attempt to determine during evaluation")

    logger.info(f"Starting model evaluation with temperature {args.temperature}")
    results = compare_models(args.lora_path, args.temperature, args.output_file, trainer_dir=trainer_dir)
    if results:
        logger.info("Evaluation completed successfully")
        logger.info(f"Final improvement: {results['improvement']:.4f}")
        logger.info(f"Results saved to: {results['output_file']}")

        # Print all output files for clarity
        logger.info("\nSUMMARY OF OUTPUT FILES:")
        logger.info(f"Comparison results: {results['output_file']}")
        logger.info(f"Base model results: {results['base_output']}")
        logger.info(f"LoRA model results: {results['lora_output']}")

        # Find and print all log files in the eval_logs directory
        if trainer_dir:
            eval_log_dir = os.path.join(trainer_dir, "eval_logs")
            if os.path.exists(eval_log_dir):
                log_files = [f for f in os.listdir(eval_log_dir) if f.endswith(".log")]

                if log_files:
                    logger.info("\nEVALUATION LOG FILES:")
                    for log_file in log_files:
                        logger.info(f"- {os.path.join(eval_log_dir, log_file)}")
    else:
        logger.warning("Evaluation failed or was skipped")
