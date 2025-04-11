import json
import sys
from collections import defaultdict  # Use defaultdict for cleaner accumulation
from pathlib import Path

import pandas as pd

# Add project root to Python path if needed (adjust relative path as necessary)
# project_root = Path(__file__).resolve().parent.parent
# sys.path.append(str(project_root))
# from config import logger # Assuming you have a logger setup


def extract_unique_paragraphs(input_paths: list[str], output_csv_path: str) -> None:
    """Extracts unique paragraphs from specified JSONL files.

    Reads Musique JSONL files (train, dev, test), finds unique paragraphs
    (regardless of is_supporting flag), combines title and text,
    tracks source question IDs, and saves to CSV.

    Args:
        input_paths: A list of paths to the input JSONL files.
        output_csv_path: Path to save the output CSV file.
    """
    output_dir = Path(output_csv_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # Use paragraph content as key, value is the set of source question IDs
    paragraphs_data = defaultdict(set)
    print("Starting paragraph extraction (including non-supporting)...")

    for file_path in input_paths:
        print(f"Processing file: {file_path}")
        try:
            with open(file_path, "r", encoding="utf-8") as infile:
                for line_num, line in enumerate(infile, 1):
                    try:
                        data = json.loads(line)
                        main_question_id = data.get("id")
                        if not main_question_id:
                            print(f"Warning: Missing 'id' in line {line_num} of {file_path}")
                            continue

                        for p in data.get("paragraphs", []):
                            title = p.get("title", "No Title")
                            text = p.get("paragraph_text", "")
                            content = f"{title}\n{text}".strip()

                            if not content:
                                continue  # Skip empty paragraphs

                            paragraphs_data[content].add(main_question_id)

                    except json.JSONDecodeError:
                        print(f"Warning: Skipping invalid JSON in line {line_num} of {file_path}")
                    except Exception as e:
                        print(f"Warning: Error processing line {line_num} in {file_path}: {e}")
        except FileNotFoundError:
            print(f"Error: Input file not found: {file_path}")
        except Exception as e:
            print(f"Error reading file {file_path}: {e}")

    print(f"Found {len(paragraphs_data)} unique paragraphs (supporting and non-supporting).")

    # Prepare data for DataFrame
    output_list = []
    sorted_content = sorted(paragraphs_data.keys())
    for chunk_id, content in enumerate(sorted_content, 1):
        question_ids = paragraphs_data[content]
        metadata = {"source_question_ids": sorted(list(question_ids))}
        output_list.append(
            {
                "chunk_id": chunk_id,
                "content": content,
                "metadata": json.dumps(metadata),  # Store metadata as JSON string
            }
        )

    if not output_list:
        print("No paragraphs found to save.")
        return
    df = pd.DataFrame(output_list)
    try:
        df.to_csv(output_csv_path, index=False)
        print(f"Successfully saved unique paragraphs to {output_csv_path}")
    except Exception as e:
        print(f"Error saving CSV file: {e}")


if __name__ == "__main__":
    RAW_DIR = Path("data/raw")
    PROCESSED_DIR = Path("data/processed")

    input_files = [
        str(RAW_DIR / "musique_ans_v1.0_train.jsonl"),
        str(RAW_DIR / "musique_ans_v1.0_dev.jsonl"),
        str(RAW_DIR / "musique_ans_v1.0_test.jsonl"),
    ]
    output_csv = str(PROCESSED_DIR / "paragraphs.csv")

    extract_unique_paragraphs(input_files, output_csv)
