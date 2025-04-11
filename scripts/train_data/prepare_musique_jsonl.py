import json
import math  # Keep math import
import os
import re  # Import re for parsing ID
from collections import defaultdict
from pathlib import Path

# import random # No longer needed
# SEED = 42 # No longer needed
# random.seed(SEED) # No longer needed


def transform_musique_data(input_path: str, output_path: str, sample_config: dict) -> None:
    """Transforms Musique data with deterministic stratified sampling using uniform selection from sorted lists.

    Reads data, categorizes by detailed hop type, sorts categories by ID,
    selects N samples uniformly spaced from each sorted category,
    combines, sorts final list by ID, and writes to output.

    Args:
        input_path: Path to the input JSONL file.
        output_path: Path to the output JSONL file.
        sample_config: Dictionary specifying samples per detailed hop type (e.g., {"2hop": 400, "3hop1": 150, ...}).
    """
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Reading all data from {input_path} for sampling...")
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
    print(f"Read {len(all_data)} total samples with IDs.")

    # Detailed Categorization by hop type
    categorized_data = defaultdict(list)
    print("Categorizing data by detailed hop type (e.g., 3hop1, 4hop2)...")
    for data in all_data:
        q_id = data["id"]
        match = re.match(r"^(2hop|3hop[12]|4hop[123])__", q_id)
        if match:
            detailed_hop_type = match.group(1)
            categorized_data[detailed_hop_type].append(data)
        # else: # Optional: log if an ID doesn't match expected pattern
        # print(f"Warning: ID {q_id} does not match expected hop pattern.")

    # Deterministic sampling using sorting and uniform index selection
    final_sample_list = []
    total_target = sum(sample_config.values())
    print(f"Sampling deterministically via uniform selection from sorted lists to get {total_target} samples...")
    # Check if all requested hop types exist in config
    for hop_type in sample_config.keys():
        if hop_type not in categorized_data:
            print(f"Warning: Hop type '{hop_type}' requested in config but not found in data.")

    for hop_type, target_count in sample_config.items():
        available_samples = categorized_data.get(hop_type, [])
        current_count = len(available_samples)
        print(f"  {hop_type}: Found {current_count} samples, need {target_count}.")

        if current_count == 0:
            continue

        # Sort the list for this category by ID
        available_samples.sort(key=lambda x: x["id"])

        selected_samples_for_hop = []
        if current_count < target_count:
            print(f"  Warning: Not enough samples for {hop_type}. Taking all {current_count} sorted samples.")
            selected_samples_for_hop = available_samples
        else:
            # Select target_count indices spread uniformly across the available samples
            print(f"  Selecting {target_count} samples uniformly from {current_count}...")
            # Calculate indices using integer interpretation of evenly spaced points
            indices_to_take = [int(i * current_count / target_count) for i in range(target_count)]
            # Ensure uniqueness in case of rounding issues with small numbers (though unlikely here)
            indices_to_take = sorted(list(set(indices_to_take)))
            # Adjust if rounding resulted in fewer than target_count unique indices
            while len(indices_to_take) < target_count:
                # This is a fallback, shouldn't happen if current_count >= target_count
                # Add indices from the end if needed, avoiding duplicates
                next_idx = indices_to_take[-1] + 1
                if next_idx < current_count and next_idx not in indices_to_take:
                    indices_to_take.append(next_idx)
                else:  # Should not be reachable if logic is sound
                    break

            # Select samples at the calculated indices
            selected_samples_for_hop = [
                available_samples[idx] for idx in indices_to_take[:target_count]
            ]  # Ensure we take exactly target_count

        final_sample_list.extend(selected_samples_for_hop)

    print(f"Selected {len(final_sample_list)} samples in total.")

    # Sort the final combined list by ID for consistent output order
    print("Sorting the final combined sample list by ID...")
    final_sample_list.sort(key=lambda x: x["id"])

    # Process and write the selected samples
    print(f"Processing and writing {len(final_sample_list)} selected samples to {output_path}...")
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
                    unique_valid_answers = list(set(valid_answers))

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
        print(f"Successfully processed and wrote {count} samples.")
    except Exception as e:
        print(f"An unexpected error occurred during writing: {e}")


if __name__ == "__main__":
    # Define file paths
    RAW_DIR = Path("data/raw")
    PROCESSED_DIR = Path("data/processed")

    # Define detailed sampling configuration
    SAMPLING_CONFIG = {
        "2hop": 400,
        "3hop1": 150,
        "3hop2": 150,
        "4hop1": 100,
        "4hop2": 100,
        "4hop3": 100,
    }  # Total = 1000

    transform_musique_data(
        str(RAW_DIR / "musique_ans_v1.0_train.jsonl"), str(PROCESSED_DIR / "questions.jsonl"), SAMPLING_CONFIG
    )

    print(
        "\nMusique JSONL transformation and detailed deterministic sampling (uniform selection from sorted) complete."
    )
    # Note: Dev/Test files are not processed by default with this sampling logic.
