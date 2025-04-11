import json
import math  # Import math for ceiling division
import sys
import traceback  # Import traceback
from pathlib import Path

import pandas as pd

# Add project root to Python path if needed (adjust relative path as necessary)
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from src.embeddings import CustomHuggingFaceEmbeddings

# Import FAISS after potentially adding to sys.path
try:
    from langchain_community.vectorstores import FAISS
except ImportError:
    print("Error: langchain_community or FAISS not installed. Please install with 'pip install langchain faiss-cpu'")
    sys.exit(1)


def build_faiss_index_from_csv(csv_path: str, index_save_path: str, batch_size: int = 128) -> None:
    """Builds a FAISS index from a CSV containing paragraph content and metadata.

    Reads a CSV file, generates embeddings for the 'content' column in batches,
    and saves the FAISS index files (index.faiss, index.pkl) locally.

    Args:
        csv_path: Path to the input CSV file (e.g., data/processed/paragraphs.csv).
        index_save_path: Path to the directory where the index files should be saved.
        batch_size: Number of texts to process in each embedding batch.
    """
    print(f"Loading paragraphs from {csv_path}")
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: CSV file not found at {csv_path}. Please run the extraction script first.")
        return
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return

    if "content" not in df.columns or "metadata" not in df.columns:
        print("Error: CSV file must contain 'content' and 'metadata' columns.")
        return

    if df.empty:
        print("Warning: Input CSV file is empty. No index will be built.")
        return

    # Prepare documents for FAISS
    texts = df["content"].astype(str).tolist()
    metadatas = []
    try:
        metadatas = [json.loads(m) for m in df["metadata"].tolist()]
        print(f"Prepared {len(texts)} texts and {len(metadatas)} metadatas.")
    except json.JSONDecodeError as e:
        print(f"Error parsing metadata JSON: {e}. Check the format in {csv_path}")
        traceback.print_exc()  # Print traceback for JSON errors
        return
    except Exception as e:
        print(f"Error processing metadata: {e}")
        traceback.print_exc()  # Print traceback for other metadata errors
        return

    if not texts or not metadatas or len(texts) != len(metadatas):
        print(f"Error: Mismatch or empty texts/metadatas. Texts: {len(texts)}, Metadatas: {len(metadatas)}")
        return

    print("Initializing embeddings model...")
    try:
        embeddings = CustomHuggingFaceEmbeddings()
    except Exception as e:
        print(f"Error initializing embeddings model: {e}")
        traceback.print_exc()
        return
    print("Embeddings model initialized successfully.")

    vectorstore = None
    num_batches = math.ceil(len(texts) / batch_size)
    print(f"Processing {len(texts)} texts in {num_batches} batches of size {batch_size}...")

    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(texts))
        batch_texts = texts[start_idx:end_idx]
        batch_metadatas = metadatas[start_idx:end_idx]
        print(f"  Processing batch {i + 1}/{num_batches} (indices {start_idx}-{end_idx - 1})...")

        try:
            if i == 0:
                # Initialize the vector store with the first batch
                print(f"    Initializing FAISS index with first batch...")
                vectorstore = FAISS.from_texts(texts=batch_texts, embedding=embeddings, metadatas=batch_metadatas)
                print("    FAISS index initialized.")
            else:
                # Add subsequent batches to the existing store
                if vectorstore is None:
                    print("Error: vectorstore is None after first batch, cannot add more texts.")
                    return  # Should not happen if first batch succeeded
                print(f"    Adding batch {i + 1} to FAISS index...")
                vectorstore.add_texts(texts=batch_texts, metadatas=batch_metadatas)
                print(f"    Batch {i + 1} added.")

        except Exception as e:
            print(f"Error processing batch {i + 1} (indices {start_idx}-{end_idx - 1}): {e}")
            traceback.print_exc()
            print("Stopping index creation due to error in batch processing.")
            return  # Exit if any batch fails

    if vectorstore is None:
        print("Error: Failed to create or add any data to the vectorstore.")
        return

    # Save the completed index
    try:
        print(f"Attempting to save final FAISS index files to directory: {index_save_path}")
        # Ensure the target directory exists before saving
        Path(index_save_path).mkdir(parents=True, exist_ok=True)
        vectorstore.save_local(index_save_path)
        print(f"Successfully saved final FAISS index files (index.faiss, index.pkl) to: {index_save_path}")
    except Exception as e:
        print(f"Error during final vectorstore.save_local to {index_save_path}: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    # Define paths relative to this script or use absolute paths
    PROCESSED_DIR = Path("data/processed")
    INPUT_CSV = str(PROCESSED_DIR / "paragraphs.csv")
    # FAISS save_local will save index.faiss and index.pkl in this directory
    INDEX_SAVE_DIR = str(PROCESSED_DIR)  # Save directly to processed dir

    build_faiss_index_from_csv(INPUT_CSV, INDEX_SAVE_DIR, batch_size=128)
