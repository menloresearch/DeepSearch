import json
import sys
from pathlib import Path

import pandas as pd

# Add project root to Python path for imports
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

# Assuming these are defined in your project structure
from config import DATA_DIR, logger  # Adjust import as needed
from src.embeddings import CustomHuggingFaceEmbeddings

# Import FAISS after potentially adding to sys.path
try:
    from langchain_community.vectorstores import FAISS

    faiss_installed = True
except ImportError:
    print("Warning: langchain_community or faiss not installed. Cannot check FAISS index.")
    faiss_installed = False


def check_output_files(processed_dir: Path):
    """Prints head and tail of key processed files and FAISS index info.

    Args:
        processed_dir: The path to the 'data/processed' directory.
    """
    print("--- Checking Processed Files ---")

    # 1. Check paragraphs.csv
    csv_path = processed_dir / "paragraphs.csv"
    print(f"\n--- Checking {csv_path} ---")
    try:
        df = pd.read_csv(csv_path)
        print("First 3 rows:")
        print(df.head(3).to_string())
        print("\nLast 3 rows:")
        print(df.tail(3).to_string())
        print(f"Total rows: {len(df)}")
    except FileNotFoundError:
        print(f"Error: {csv_path} not found.")
    except Exception as e:
        print(f"Error reading {csv_path}: {e}")

    # 2. Check questions.jsonl
    jsonl_path = processed_dir / "questions.jsonl"
    print(f"\n--- Checking {jsonl_path} ---")
    try:
        with open(jsonl_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        num_lines = len(lines)
        print(f"Total lines: {num_lines}")

        if num_lines > 0:
            print("\nFirst 3 lines (parsed JSON):")
            for i in range(min(3, num_lines)):
                try:
                    print(json.loads(lines[i].strip()))
                except json.JSONDecodeError:
                    print(f"  (Error parsing line {i + 1})")

        if num_lines > 3:
            print("\nLast 3 lines (parsed JSON):")
            for i in range(max(0, num_lines - 3), num_lines):
                try:
                    print(json.loads(lines[i].strip()))
                except json.JSONDecodeError:
                    print(f"  (Error parsing line {i + 1})")
        elif num_lines > 0:
            print("\n(Less than 6 lines total, showing all)")

    except FileNotFoundError:
        print(f"Error: {jsonl_path} not found.")
    except Exception as e:
        print(f"Error reading {jsonl_path}: {e}")

    # 3. Check FAISS index
    print(f"\n--- Checking FAISS Index in {processed_dir} ---")
    if not faiss_installed:
        print("Skipping FAISS check as required libraries are not installed.")
        return

    # FAISS loads from the directory containing index.faiss and index.pkl
    index_dir = processed_dir
    index_file = index_dir / "index.faiss"
    pkl_file = index_dir / "index.pkl"

    if not index_file.exists() or not pkl_file.exists():
        print(f"Error: FAISS index files (index.faiss, index.pkl) not found in {index_dir}")
        return

    try:
        print("Initializing embeddings model for loading index...")
        embeddings = CustomHuggingFaceEmbeddings()
        print("Loading FAISS index...")
        # FAISS.load_local requires the folder_path and the embeddings object
        vectorstore = FAISS.load_local(str(index_dir), embeddings, allow_dangerous_deserialization=True)
        print("FAISS index loaded successfully.")
        # Access the underlying FAISS index object to get the total number of vectors
        print(f"Total vectors in index: {vectorstore.index.ntotal}")
    except Exception as e:
        print(f"Error loading or checking FAISS index from {index_dir}: {e}")
        import traceback

        traceback.print_exc()

    print("\n--- Check Complete ---")


if __name__ == "__main__":
    # Assuming the script is run from the project root or paths are relative
    PROCESSED_PATH = Path("data/processed")
    check_output_files(PROCESSED_PATH)
