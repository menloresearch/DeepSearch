"""
Evaluate model performance using vLLM and unsloth.

This script evaluates the performance of a model using vLLM for fast inference
and unsloth for LoRA support.
"""

import argparse
import os
import time
from datetime import datetime

from unsloth import FastLanguageModel
from vllm import SamplingParams

from src import (
    apply_chat_template,
    build_reward_correctness_fn,
    build_user_prompt,
    get_qa_dataset,
    get_system_prompt,
    run_eval,
)
from config import MODEL_NAME, logger


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


def get_sampling_params(temperature=0.5):
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
        lora_path: Path to LoRA weights (None for base model)
        temperature: Sampling temperature
        output_file: File to write results to
        trainer_dir: Directory containing the checkpoints
    """
    sampling_params = get_sampling_params(temperature=temperature)

    # Set up output directory
    if trainer_dir:
        eval_log_dir = os.path.join(trainer_dir, "eval_logs")
    else:
        eval_log_dir = "eval_logs"
    os.makedirs(eval_log_dir, exist_ok=True)

    # Create file names based on model type
    model_prefix = "lora" if lora_path else "base"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Define all output file paths
    eval_log_file = os.path.join(eval_log_dir, f"{model_prefix}_model_eval_{timestamp}.log")
    output_file = os.path.join(eval_log_dir, f"{model_prefix}_model_results.txt")
    debug_file = os.path.join(eval_log_dir, f"{model_prefix}_model_results_debug.json")

    logger.info(f"Writing evaluation log to: {eval_log_file}")
    logger.info(f"Results will be saved to: {output_file}")

    # Function to generate completions using agentic approach
    def eval_generate_fn(inputs):
        start_time = time.time()

        # Format inputs as chat messages with system prompt
        messages = [
            {
                "messages": [
                    {"role": "system", "content": get_system_prompt()},
                    {"role": "user", "content": build_user_prompt(input_text)},
                ]
            }
            for input_text in inputs
        ]

        if lora_path:
            lora_request = model.load_lora(lora_path)
            load_time = time.time() - start_time
            logger.info(f"LoRA adapter loaded in {load_time:.2f} seconds: {lora_request}")
            responses = model.fast_generate(
                [apply_chat_template(msg, tokenizer=tokenizer)["text"] for msg in messages],
                sampling_params=sampling_params,
                lora_request=lora_request,
            )
        else:
            responses = model.fast_generate(
                [apply_chat_template(msg, tokenizer=tokenizer)["text"] for msg in messages],
                sampling_params=sampling_params,
            )

        gen_time = time.time() - start_time
        logger.debug(f"Generation completed in {gen_time:.2f} seconds")
        return responses

    def verifier_generate_fn(inputs):
        # Use a lower temperature for verification to get more consistent results
        verifier_params = get_sampling_params(temperature=0.1)

        # Format inputs as chat messages with system prompt
        messages = [
            {
                "messages": [
                    {"role": "system", "content": get_system_prompt()},
                    {"role": "user", "content": build_user_prompt(input_text)},
                ]
            }
            for input_text in inputs
        ]

        return model.fast_generate(
            [apply_chat_template(msg, tokenizer=tokenizer)["text"] for msg in messages],
            sampling_params=verifier_params,
        )

    # Prepare the verification function
    verify_fn = build_reward_correctness_fn(verifier_generate_fn, tokenizer)

    # Get the dataset and prepare questions and answers
    train_dataset, test_dataset = get_qa_dataset()
    questions = test_dataset["prompt"]
    inputs = questions

    logger.info(f"Verifying {len(inputs)} answers...")

    # Run the evaluation
    start_time = time.time()
    model_type = "LoRA" if lora_path else "Base"
    logger.info(f"Starting {model_type} model evaluation...")

    # Run evaluation using the agentic approach
    full_chat_states = run_eval(
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
        lora_path: Path to LoRA weights
        temperature: Sampling temperature
        output_file: File to write results to (optional)
        trainer_dir: Directory containing the trainer output
    """
    # Set up output directory
    if trainer_dir:
        eval_log_dir = os.path.join(trainer_dir, "eval_logs")
    else:
        eval_log_dir = "eval_logs"
    os.makedirs(eval_log_dir, exist_ok=True)

    # Define the comparison file path if not provided
    if output_file is None:
        output_file = os.path.join(eval_log_dir, "model_comparison_results.txt")

    # Define file paths for individual model results
    base_output = os.path.join(eval_log_dir, "base_model_results.txt")
    lora_output = os.path.join(eval_log_dir, "lora_model_results.txt")

    model, tokenizer = setup_model_and_tokenizer()

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
        default="trainer_output_example/checkpoint-101",
        help="Path to LoRA weights",
    )
    parser.add_argument("--temperature", type=float, default=0.5, help="Sampling temperature")
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="File to write results to (optional)",
    )
    parser.add_argument(
        "--trainer_dir",
        type=str,
        default=None,
        help="Directory containing the trainer output",
    )
    args = parser.parse_args()

    logger.info(f"Starting model evaluation with temperature {args.temperature}")
    results = compare_models(args.lora_path, args.temperature, args.output_file, trainer_dir=args.trainer_dir)
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
        eval_log_dir = os.path.join(args.trainer_dir, "eval_logs") if args.trainer_dir else "eval_logs"
        if os.path.exists(eval_log_dir):
            log_files = [f for f in os.listdir(eval_log_dir) if f.endswith(".log")]
            if log_files:
                logger.info("\nEVALUATION LOG FILES:")
                for log_file in log_files:
                    logger.info(f"- {os.path.join(eval_log_dir, log_file)}")
    else:
        logger.warning("Evaluation failed or was skipped")
