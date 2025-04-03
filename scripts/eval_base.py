"""Simple script to evaluate base model performance."""

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

# Add project root to Python path
project_root = str(Path(__file__).parent.parent)
sys.path.append(project_root)

from unsloth import FastLanguageModel
from vllm import SamplingParams

from src import (
    apply_chat_template,
    build_reward_correctness_fn,
    build_user_prompt,
    get_system_prompt,
    run_eval,
)
from src.config import logger


def main():
    """Run base model evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate base model")
    parser.add_argument("--model_name", type=str, required=True, help="Name/path of the model to evaluate")
    parser.add_argument("--temperature", type=float, default=0, help="Sampling temperature")
    args = parser.parse_args()

    logger.info(f"🚀 Setting up model {args.model_name}...")

    # Setup model
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_name,
        max_seq_length=4096 * 2,
        load_in_4bit=True,
        fast_inference=True,
    )

    # Setup sampling params
    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=0.95,
        max_tokens=4096,
    )

    # Setup verifier with lower temperature
    verifier_params = SamplingParams(
        temperature=0.1,  # Lower temperature for more consistent verification
        top_p=0.95,
        max_tokens=4096,
    )

    def generate_fn(inputs):
        """Generate responses for inputs."""
        messages = [
            {
                "messages": [
                    {"role": "system", "content": get_system_prompt()},
                    {"role": "user", "content": build_user_prompt(input_text)},
                ]
            }
            for input_text in inputs
        ]

        outputs = model.fast_generate(
            [apply_chat_template(msg, tokenizer=tokenizer)["text"] for msg in messages],
            sampling_params=sampling_params,
        )
        return outputs

    def verifier_generate_fn(inputs):
        """Generate verification responses with lower temperature."""
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

    # Build verifier
    verify_fn = build_reward_correctness_fn(verifier_generate_fn, tokenizer)

    # Setup output directories
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    eval_log_dir = "eval_logs"
    os.makedirs(eval_log_dir, exist_ok=True)

    output_file = os.path.join(eval_log_dir, f"base_model_results_{timestamp}.txt")
    debug_file = os.path.join(eval_log_dir, f"base_model_debug_{timestamp}.json")

    logger.info("📝 Starting evaluation...")
    logger.info(f"Results will be saved to: {output_file}")
    logger.info(f"Debug info will be saved to: {debug_file}")

    # Run evaluation using the agentic approach
    full_chat_states = run_eval(
        generate_fn=generate_fn,
        verify_fn=verify_fn,
        tokenizer=tokenizer,
        output_file=output_file,
        debug_file=debug_file,
    )

    logger.info("✨ Evaluation completed!")
    logger.info(f"Check {output_file} for detailed results")


if __name__ == "__main__":
    main()
