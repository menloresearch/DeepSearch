"""Simple script to evaluate LoRA model performance."""

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
    """Run LoRA model evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate LoRA model")
    parser.add_argument("--model_name", type=str, required=True, help="Name/path of the base model")
    parser.add_argument("--lora_path", type=str, required=True, help="Path to LoRA weights")
    args = parser.parse_args()

    print(f"🚀 Setting up model {args.model_name} with LoRA from {args.lora_path}...")

    # Setup model with LoRA support
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_name,
        max_seq_length=4096 * 2,
        load_in_4bit=True,
        fast_inference=True,
        max_lora_rank=64,
    )

    # Setup LoRA
    model = FastLanguageModel.get_peft_model(
        model,
        r=64,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_alpha=64,
        use_gradient_checkpointing=True,
        random_state=3407,
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

        lora_request = model.load_lora(args.lora_path)
        outputs = model.fast_generate(
            [apply_chat_template(msg, tokenizer=tokenizer)["text"] for msg in messages],
            sampling_params=sampling_params,
            lora_request=lora_request,
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
    print("🎯 LORA MODEL EVALUATION RESULTS:")
    print(f"{'=' * 50}")
    print(f"✨ Base Model: {args.model_name}")
    print(f"🔧 LoRA Path: {args.lora_path}")
    print(f"📊 Accuracy: {accuracy:.4f} ({sum(rewards)}/{len(rewards)} correct)")


if __name__ == "__main__":
    main()
