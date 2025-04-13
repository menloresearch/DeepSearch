"""Prepares a deterministic sampled dev set (questions_dev.jsonl) from raw Musique dev data."""

import json
import math
import os
import re
from collections import defaultdict
from pathlib import Path


def transform_musique_dev_data(input_path: str, output_path: str, sample_config: dict) -> None:
    """Transforms Musique dev data with deterministic stratified sampling using uniform selection from sorted lists.

    Reads dev data, categorizes by hop type (2, 3, 4), sorts categories by ID,
    selects N samples uniformly spaced from each sorted category based on sample_config,
    combines, sorts final list by ID, combines answers/aliases, extracts supporting paras,
    and writes the transformed data to output_path.

    Args:
        input_path: Path to the input JSONL file (e.g., data/raw/musique_ans_v1.0_dev.jsonl).
        output_path: Path to the output JSONL file (e.g., data/processed/questions_dev.jsonl).
        sample_config: Dictionary specifying samples per hop type (e.g., {"2hop": 20, "3hop": 15, "4hop": 15}).
    """
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Reading all data from {input_path} for dev sampling...")
    all_data = []
    try:
        with open(input_path, "r", encoding="utf-8") as infile:
            for line_num, line in enumerate(infile, 1):
                try:
                    data = json.loads(line)
                    if "id" in data:
                        all_data.append(data)
                    else:
                        print(f"Warning: Skipping line {line_num} due to missing 'id' field in {input_path}")
                except json.JSONDecodeError:
                    print(f"Warning: Skipping invalid JSON in line {line_num} of {input_path}")
    except FileNotFoundError:
        print(f"Error: Input file not found at {input_path}")
        return
    except Exception as e:
        print(f"Error reading file {input_path}: {e}")
        return
    print(f"Read {len(all_data)} total samples from dev set.")

    # Categorize data by hop count (2hop, 3hop, 4hop)
    categorized_data = defaultdict(list)
    print("Categorizing data by hop type (2, 3, 4)...")
    for data in all_data:
        q_id = data["id"]
        hop_type = None
        if q_id.startswith("2hop"):
            hop_type = "2hop"
        elif q_id.startswith("3hop"):
            hop_type = "3hop"
        elif q_id.startswith("4hop"):
            hop_type = "4hop"

        if hop_type:
            categorized_data[hop_type].append(data)

    # Deterministic sampling using sorting and uniform index selection
    final_sample_list = []
    total_target = sum(sample_config.values())
    print(f"Sampling deterministically via uniform selection from sorted lists to get {total_target} dev samples...")

    for hop_type, target_count in sample_config.items():
        available_samples = categorized_data.get(hop_type, [])
        current_count = len(available_samples)
        print(f"  {hop_type}: Found {current_count} samples, need {target_count}.")

        if current_count == 0:
            continue

        available_samples.sort(key=lambda x: x["id"])
        selected_samples_for_hop = []
        if current_count < target_count:
            print(f"  Warning: Not enough samples for {hop_type}. Taking all {current_count} sorted samples.")
            selected_samples_for_hop = available_samples
        elif target_count > 0:  # Ensure target_count is positive before selecting
            print(f"  Selecting {target_count} samples uniformly from {current_count}...")
            # Calculate indices using integer interpretation of evenly spaced points
            indices_to_take = [
                int(i * (current_count - 1) / (target_count - 1)) if target_count > 1 else 0
                for i in range(target_count)
            ]  # Adjust index calc for edges
            indices_to_take = sorted(list(set(indices_to_take)))  # Ensure unique indices
            # Simple fallback if uniqueness reduced count below target
            while len(indices_to_take) < target_count and len(indices_to_take) < current_count:
                next_val = indices_to_take[-1] + 1
                if next_val < current_count:
                    indices_to_take.append(next_val)
                else:  # Cannot add more unique indices
                    break
            selected_samples_for_hop = [
                available_samples[idx] for idx in indices_to_take[:target_count]
            ]  # Select based on unique indices, capped at target

        final_sample_list.extend(selected_samples_for_hop)

    print(f"Selected {len(final_sample_list)} dev samples in total.")

    # Sort the final combined list by ID for consistent output order
    print("Sorting the final combined dev sample list by ID...")
    final_sample_list.sort(key=lambda x: x["id"])

    # Process and write the selected samples
    print(f"Processing and writing {len(final_sample_list)} selected dev samples to {output_path}...")
    count = 0
    try:
        with open(output_path, "w", encoding="utf-8") as outfile:
            for data in final_sample_list:
                try:
                    supporting_paragraphs = [
                        p["paragraph_text"] for p in data.get("paragraphs", []) if p.get("is_supporting", False)
                    ]
                    main_answer = data.get("answer", "")
                    aliases = data.get("answer_aliases", [])
                    all_answers = [main_answer] + (aliases if isinstance(aliases, list) else [])
                    valid_answers = [str(ans).strip() for ans in all_answers if ans and str(ans).strip()]
                    unique_valid_answers = list(set(valid_answers))  # Keep unique, don't sort alphabetically
                    combined_answer_str = " OR ".join(unique_valid_answers)

                    output_data = {
                        "id": data.get("id"),
                        "question": data.get("question"),
                        "answer": combined_answer_str,
                        "supporting_paragraphs": supporting_paragraphs,
                    }
                    outfile.write(json.dumps(output_data) + "\n")
                    count += 1
                except KeyError as e:
                    print(f"Skipping sample due to missing key {e}: {data.get('id')}")
        print(f"Successfully processed and wrote {count} dev samples.")
    except Exception as e:
        print(f"An unexpected error occurred during writing: {e}")


if __name__ == "__main__":
    # Define file paths relative to the project root
    # Ensure this script is run from the project root or adjust paths accordingly
    RAW_DIR = Path("data/raw")
    PROCESSED_DIR = Path("data/processed")

    # Define sampling configuration for the dev set
    DEV_SAMPLING_CONFIG = {"2hop": 20, "3hop": 15, "4hop": 15}  # Total = 50

    INPUT_FILE = RAW_DIR / "musique_ans_v1.0_dev.jsonl"
    OUTPUT_FILE = PROCESSED_DIR / "questions_dev.jsonl"

    transform_musique_dev_data(str(INPUT_FILE), str(OUTPUT_FILE), DEV_SAMPLING_CONFIG)

    print(f"\nMusique DEV JSONL transformation and deterministic sampling complete.")
