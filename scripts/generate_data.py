"""
This script performs two main tasks:
1. It loads a markdown document, splits it into chunks, generates embeddings,
   and builds a FAISS index with the original and paraphrased chunks (which is saved locally).
2. It generates QA pairs from the document using llama.
   For each chunk (using a sliding window for context), it generates multiple question-answer pairs
   with different difficulties. The generation is performed in batch with one retry for failed prompts.
   Successfully generated QA pairs are saved to "data/questions.jsonl".
"""

import json
import re
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

import pandas as pd
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_community.vectorstores import FAISS
from unsloth import FastLanguageModel
from vllm import SamplingParams

from config import DATA_DIR, logger
from src.embeddings import CustomHuggingFaceEmbeddings

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


# ========= Part 2: QA Generation using Llama Backend =========

# Setup Llama backend via unsloth and vLLM

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

# Define paraphrasing styles and parameters
PARAPHRASE_PROMPTS = [
    """Rewrite this text in a formal, scholarly tone. Keep it very concise - summarize in 1-2 short sentences. Only output the paraphrased text:

    TEXT: {text}""",
    """Rewrite this text in a clear, simple way that's easy to understand. Provide a medium-length explanation with key details. Only output the paraphrased text:

    TEXT: {text}""",
    """Rewrite this text in a vivid, engaging style. Expand on the details and provide a comprehensive, detailed version. Only output the paraphrased text:

    TEXT: {text}""",
]

# Sampling parameters for different lengths
sampling_params_short = SamplingParams(
    temperature=0.3,
    top_p=0.95,
    max_tokens=64,  # Short responses
)

sampling_params_medium = SamplingParams(
    temperature=0.3,
    top_p=0.95,
    max_tokens=256,  # Medium responses
)

sampling_params_long = SamplingParams(
    temperature=0.3,
    top_p=0.95,
    max_tokens=512,  # Long responses
)


def generate_paraphrases(text: str) -> list:
    """
    Generate three different paraphrased versions with varying lengths.

    Args:
        text: Text to paraphrase

    Returns:
        List of three paraphrased versions (short, medium, long)
    """
    responses = []
    sampling_params_list = [
        sampling_params_short,
        sampling_params_medium,
        sampling_params_long,
    ]

    for prompt_template, sampling_params in zip(PARAPHRASE_PROMPTS, sampling_params_list):
        formatted_prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt_template.format(text=text)}],
            tokenize=False,
            add_generation_prompt=True,
        )

        output = model.fast_generate([formatted_prompt], sampling_params=sampling_params)
        responses.append(output[0].outputs[0].text)

    return responses


# Paraphrase all chunks and add to vector store
logger.info("Paraphrasing chunks and adding to vector store...")
all_paraphrased = []
chunk_ids = []

for i, chunk in enumerate(chunks):
    # Get paragraphs from chunk
    paragraphs = [p.strip() for p in chunk.page_content.split("\n\n") if p.strip()]

    for paragraph in paragraphs:
        # Generate 3 paraphrased versions
        paraphrased_versions = generate_paraphrases(paragraph)

        # Save original paragraph ID for reference
        for version in paraphrased_versions:
            all_paraphrased.append({"chunk_id": i + 1, "original_paragraph": paragraph, "paraphrased_text": version})

# Save paraphrased chunks to CSV for inspection
paraphrased_df = pd.DataFrame(all_paraphrased)
paraphrased_csv_path = DATA_DIR / "paragraphs_noise.csv"
paraphrased_df.to_csv(paraphrased_csv_path, index=False)
logger.info(f"Saved {len(all_paraphrased)} paraphrased paragraphs to {paraphrased_csv_path}")


paraphrased_docs = [
    Document(page_content=item["paraphrased_text"], metadata={"chunk_id": item["chunk_id"], "is_paraphrase": True})
    for item in all_paraphrased
]

# Process embeddings in smaller batches to avoid OOM
logger.info(f"Creating FAISS index with {len(paraphrased_docs)} documents in batches")
batch_size = 100  # Process 100 documents at a time
paraphrased_vectorstore = None

for i in range(0, len(paraphrased_docs), batch_size):
    batch = paraphrased_docs[i : i + batch_size]
    logger.info(f"Processing batch {i // batch_size + 1}/{(len(paraphrased_docs) + batch_size - 1) // batch_size}")

    # Create a new FAISS index for this batch
    batch_vectorstore = FAISS.from_documents(batch, embeddings)

    # Merge with existing index or create a new one
    if paraphrased_vectorstore is None:
        paraphrased_vectorstore = batch_vectorstore
    else:
        paraphrased_vectorstore.merge_from(batch_vectorstore)

# Merge with main vectorstore
if paraphrased_vectorstore is not None:
    vectorstore.merge_from(paraphrased_vectorstore)
    logger.info(f"Updated FAISS index with {len(paraphrased_docs)} paraphrased paragraphs")

    # Save the updated vector store
    vectorstore.save_local(str(DATA_DIR))
    logger.info(f"Saved updated FAISS index to {DATA_DIR}")
else:
    logger.warning("No paraphrased documents were processed successfully")


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


def generate_question_batch_for_chunks(chunks: list, num_questions: int = 2, difficulty=None) -> list:
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
                    logger.warning(f"Retry failed for chunk {idx + 1}: not enough valid QA pairs")
            else:
                results[idx] = None
                logger.warning(f"Retry failed for chunk {idx + 1}: parsing failed")

    # Build final output, skipping prompts that failed even after retry
    final_questions = []
    for i, qa_list in enumerate(results):
        if qa_list is not None:
            for qa in qa_list:
                # Get supporting paragraphs by splitting chunk content into paragraphs
                supporting_paragraphs = [p.strip() for p in chunk_contents[i].split("\n\n") if p.strip()]

                final_questions.append(
                    {
                        "id": str(chunk_ids[i]),
                        "question": qa[0],
                        "answer": qa[1],
                        "supporting_paragraphs": supporting_paragraphs,
                    }
                )

    logger.info(f"Generated {len(final_questions)} valid QA pairs")
    return final_questions


# Generate QA pairs in batch (using a sliding window over the chunks)
logger.info("Generating question-answer pairs...")
all_questions = generate_question_batch_for_chunks(chunks, num_questions=2, difficulty="medium")
logger.info(f"Generated {len(all_questions)} QA pairs.")

# Save the QA pairs to a JSONL file
questions_path = DATA_DIR / "questions.jsonl"
with open(questions_path, "w") as f:
    for question in all_questions:
        f.write(json.dumps(question) + "\n")
logger.info(f"Saved questions to {questions_path}")
