"""Simple script to evaluate base model performance."""

import argparse
import sys
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
    get_qa_dataset,
    get_system_prompt,
)


def main():
    """Run base model evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate base model")
    parser.add_argument("--model_name", type=str, required=True, help="Name/path of the model to evaluate")
    args = parser.parse_args()

    print(f"🚀 Setting up model {args.model_name}...")

    # Setup model
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_name,
        max_seq_length=4096 * 2,
        load_in_4bit=True,
        fast_inference=True,
    )

    # Setup sampling params
    sampling_params = SamplingParams(
        temperature=0.5,
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

        # Format outputs as chat messages
        formatted_outputs = []
        for output in outputs:
            formatted_outputs.append(
                {
                    "messages": [
                        {"role": "system", "content": get_system_prompt()},
                        {"role": "assistant", "content": output.outputs[0].text},
                    ]
                }
            )
        return formatted_outputs

    # Get dataset
    _, test_dataset = get_qa_dataset()
    questions = test_dataset["prompt"]
    answers = test_dataset["answer"]

    print(f"📝 Evaluating {len(questions)} questions...")

    # Build verifier
    verify_fn = build_reward_correctness_fn(generate_fn, tokenizer)

    # Run evaluation
    completions = generate_fn(questions)
    rewards = verify_fn(questions, completions, answer=answers)
    accuracy = sum(rewards) / len(rewards)

    print(f"\n{'=' * 50}")
    print("🎯 BASE MODEL EVALUATION RESULTS:")
    print(f"{'=' * 50}")
    print(f"✨ Model: {args.model_name}")
    print(f"📊 Accuracy: {accuracy:.4f} ({sum(rewards)}/{len(rewards)} correct)")


if __name__ == "__main__":
    main()
