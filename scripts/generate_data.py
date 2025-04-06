"""
This script performs two main tasks:
1. It loads a markdown document, splits it into chunks, generates embeddings,
   and builds a FAISS index (which is saved locally).
2. It generates QA pairs from the document using llama.
   For each chunk (using a sliding window for context), it generates multiple question-answer pairs
   with different difficulties. The generation is performed in batch with one retry for failed prompts.
   Successfully generated QA pairs are saved to "data/questions.json".

Requirements:
    pip install langchain faiss-cpu unsloth vllm
"""

import json
import re
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter

# ========= Part 1: Document Processing and Embedding Generation =========
# Load and split the markdown document using LangChain
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_community.vectorstores import FAISS

from config import DATA_DIR, logger
from src.deepsearch.embeddings import CustomHuggingFaceEmbeddings

# Load your markdown file (adjust the path as needed)
loader = UnstructuredMarkdownLoader("./data/mission_report.md")
docs = loader.load()

# Split the document into smaller chunks (each 1000 characters, no overlap)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
chunks = text_splitter.split_documents(docs)

# Save chunks to CSV for easy inspection
chunks_df = pd.DataFrame(
    {
        "chunk_id": range(1, len(chunks) + 1),
        "content": [chunk.page_content for chunk in chunks],
        "metadata": [chunk.metadata for chunk in chunks],
    }
)
chunks_df.to_csv(DATA_DIR / "chunks.csv", index=False)
logger.info(f"Saved {len(chunks)} chunks to {DATA_DIR}/chunks.csv")

embeddings = CustomHuggingFaceEmbeddings()

# Create a FAISS vector store from the document chunks and save it locally
vectorstore = FAISS.from_documents(chunks, embeddings)
vectorstore.save_local(str(DATA_DIR))
logger.info(f"Saved FAISS index to {DATA_DIR}")

# TODO: add the paraphrased chunks to the vector store

# ========= Part 2: QA Generation using Llama Backend =========

# Setup Llama backend via unsloth and vLLM
from unsloth import FastLanguageModel
from vllm import SamplingParams

# Load the Llama model (adjust parameters as needed)
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="meta-llama/meta-Llama-3.1-8B-Instruct",
    max_seq_length=4096,
    load_in_4bit=True,  # Use 4-bit quantization if desired
    fast_inference=True,  # Enable fast inference
    gpu_memory_utilization=0.6,  # Adjust based on your GPU memory
)

# Define sampling parameters for generation
sampling_params = SamplingParams(
    temperature=0.3,
    top_p=0.95,
    max_tokens=4096,
)


def batch_generate(prompts: list) -> list:
    """
    Given a list of prompt strings, returns a list of generated outputs.
    """

    def format_input(text: str) -> str:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": text}],
            tokenize=False,
            add_generation_prompt=True,
        )

    formatted = [format_input(p) for p in prompts]
    outputs = model.fast_generate(formatted, sampling_params=sampling_params)
    return [output.outputs[0].text for output in outputs]


def parse_qa_block(block: str):
    """
    Parses a QA block that should contain exactly three non-empty lines:
      - A line starting with "Question:"
      - A line starting with "Answer:"
      - A line starting with "Difficulty:"

    If the markers are not present but the block contains exactly three lines,
    those are used in order.

    Returns a tuple (question, answer, difficulty) or None if parsing fails.
    """
    lines = [line.strip() for line in block.splitlines() if line.strip()]
    if not lines:
        return None

    question, answer, difficulty = None, None, None
    for line in lines:
        lower = line.lower()
        if question is None and lower.startswith("question:"):
            question = line[len("question:") :].strip()
        elif answer is None and lower.startswith("answer:"):
            answer = line[len("answer:") :].strip()
        elif difficulty is None and lower.startswith("difficulty:"):
            difficulty = line[len("difficulty:") :].strip()

    if question and answer and difficulty:
        return question, answer, difficulty
    if len(lines) == 3:
        return lines[0], lines[1], lines[2]
    return None


def parse_multiple_qa_output(output: str) -> list:
    """
    Splits the output into blocks (separated by one or more blank lines) and
    attempts to parse each as a QA pair.

    Returns a list of successfully parsed QA tuples.
    """
    blocks = re.split(r"\n\s*\n", output.strip())
    qa_pairs = []
    for block in blocks:
        parsed = parse_qa_block(block)
        if parsed:
            qa_pairs.append(parsed)
    return qa_pairs


def generate_question_batch_for_chunks(
    chunks: list, num_questions: int = 2, difficulty=None
) -> list:
    """
    Generates QA pairs for multiple chunks in batch.

    For each chunk, generates questions based on its content only.
    Each prompt instructs the model to output exactly three lines per QA pair with markers.
    Failed prompts are retried once in batch; if still unsuccessful, they are skipped.

    Returns a list of dicts with keys: "chunk_id", "question", "answer", "difficulty", "chunk_content".
    """
    prompts = []
    chunk_ids = []
    chunk_contents = []

    # Prepare prompts for each chunk
    for i, chunk in enumerate(chunks):
        current = chunk.page_content
        prompt = (
            f"You are a question generator. Generate {num_questions} questions based on the following text.\n"
            "Rules:\n"
            "1. Questions must be answerable using ONLY the information in the text\n"
            "2. Answers must be directly stated in the text\n"
            "3. Each question should test understanding of a different aspect of the text\n"
            "4. Questions should be clear and specific\n"
            "5. Answers should be concise and factual\n\n"
            "For each QA pair, output exactly three lines with no extra commentary:\n"
            "Line 1: Question: <your question>\n"
            "Line 2: Answer: <the answer>\n"
            "Line 3: Difficulty: <easy, medium, or hard>\n"
            "Do not include any additional text.\n\n"
            "Text:\n"
            f"{current}\n"
        )
        prompts.append(prompt)
        chunk_ids.append(i + 1)  # 1-based indexing
        chunk_contents.append(current)

    # First batch generation
    outputs = batch_generate(prompts)
    results = []
    for _ in range(len(outputs)):
        results.append(None)
    failed_indices = []

    # Parse each output
    for idx, output in enumerate(outputs):
        qa_pairs = parse_multiple_qa_output(output)
        if qa_pairs is None or len(qa_pairs) < num_questions:
            failed_indices.append(idx)
            logger.warning(f"Failed to generate enough QA pairs for chunk {idx + 1}")
        else:
            # Validate that answers exist in chunk content
            valid_pairs = []
            for q, a, d in qa_pairs:
                if a.lower() in chunk_contents[idx].lower():
                    valid_pairs.append((q, a, d))
                else:
                    logger.warning(f"Answer not found in chunk content: {a}")

            if len(valid_pairs) >= num_questions:
                results[idx] = valid_pairs[:num_questions]
            else:
                failed_indices.append(idx)
                logger.warning(f"Not enough valid QA pairs for chunk {idx + 1}")

    # Retry failed prompts in batch
    if failed_indices:
        logger.info(f"Retrying {len(failed_indices)} failed prompt(s)...")
        retry_prompts = [prompts[i] for i in failed_indices]
        retry_outputs = batch_generate(retry_prompts)
        for j, idx in enumerate(failed_indices):
            qa_pairs = parse_multiple_qa_output(retry_outputs[j])
            if qa_pairs is not None and len(qa_pairs) >= num_questions:
                # Validate answers again
                valid_pairs = []
                for q, a, d in qa_pairs:
                    if a.lower() in chunk_contents[idx].lower():
                        valid_pairs.append((q, a, d))

                if len(valid_pairs) >= num_questions:
                    results[idx] = valid_pairs[:num_questions]
                else:
                    results[idx] = None
                    logger.warning(
                        f"Retry failed for chunk {idx + 1}: not enough valid QA pairs"
                    )
            else:
                results[idx] = None
                logger.warning(f"Retry failed for chunk {idx + 1}: parsing failed")

    # Build final output, skipping prompts that failed even after retry
    final_questions = []
    for i, qa_list in enumerate(results):
        if qa_list is not None:
            for qa in qa_list:
                final_questions.append(
                    {
                        "chunk_id": chunk_ids[i],
                        "question": qa[0],
                        "answer": qa[1],
                        "difficulty": qa[2],
                        "chunk_content": chunk_contents[i],
                    }
                )

    logger.info(f"Generated {len(final_questions)} valid QA pairs")
    return final_questions


# Generate QA pairs in batch (using a sliding window over the chunks)
all_questions = generate_question_batch_for_chunks(
    chunks, num_questions=2, difficulty="medium"
)
logger.info(f"Generated {len(all_questions)} QA pairs.")

# Save the QA pairs to a JSON file
questions_path = DATA_DIR / "questions.json"
with open(questions_path, "w") as f:
    json.dump(all_questions, f, indent=2)
logger.info(f"Saved questions to {questions_path}")
