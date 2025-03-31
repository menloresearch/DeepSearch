"""
Simple script to load unsloth checkpoint and save to FP16 format.
"""

import os

from unsloth import FastLanguageModel


def load_model(
    model_name: str,
    max_seq_length: int = 8192,
    load_in_4bit: bool = True,
    fast_inference: bool = True,
    max_lora_rank: int = 64,
    gpu_memory_utilization: float = 0.6,
):
    """Load model and tokenizer with unsloth."""
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        load_in_4bit=load_in_4bit,
        fast_inference=fast_inference,
        max_lora_rank=max_lora_rank,
        gpu_memory_utilization=gpu_memory_utilization,
    )
    return model, tokenizer


def save_to_fp16(checkpoint_dir: str, output_dir: str | None = None):
    """
    Load unsloth checkpoint and save to FP16 format.

    Args:
        checkpoint_dir: Directory containing the checkpoint
        output_dir: Directory to save the FP16 model (default: model_merged_16bit in parent of checkpoint_dir)
    """
    if output_dir is None:
        # Get parent directory of checkpoint and create model_merged_16bit there
        parent_dir = os.path.dirname(checkpoint_dir)
        output_dir = os.path.join(parent_dir, "model_merged_16bit")

    # Load model and tokenizer
    print(f"Loading model from {checkpoint_dir}")
    model, tokenizer = load_model(checkpoint_dir)

    # Save to FP16
    print(f"Saving model to FP16 in {output_dir}")
    model.save_pretrained_merged(
        output_dir,
        tokenizer,
        save_method="merged_16bit",
    )
    print("Done!")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Save unsloth checkpoint to FP16")
    parser.add_argument(
        "checkpoint_dir",
        nargs="?",
        default="trainer_output_example/checkpoint-101",
        help="Directory containing the checkpoint (default: trainer_output_example/checkpoint-101)",
    )
    parser.add_argument(
        "--output_dir",
        help="Directory to save the FP16 model (default: model_merged_16bit in parent of checkpoint_dir)",
    )
    args = parser.parse_args()

    save_to_fp16(args.checkpoint_dir, args.output_dir)
