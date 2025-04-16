"""
Search module for RL training loop.
This module provides functions to search through vectorized documents and retrieve question-answer pairs.
"""

import json
import random

from datasets import Dataset
from langchain_community.vectorstores import FAISS

from config import DATA_DIR, logger
from src.embeddings import CustomHuggingFaceEmbeddings

PROCESSED_DATA_DIR = DATA_DIR


# Load pre-saved vectorstore
def load_vectorstore():
    """Load the pre-saved FAISS index"""
    try:
        embeddings = CustomHuggingFaceEmbeddings()
        # Load the FAISS index from the data directory
        logger.info(f"Loading FAISS index from: {PROCESSED_DATA_DIR}")
        vectorstore = FAISS.load_local(str(PROCESSED_DATA_DIR), embeddings, allow_dangerous_deserialization=True)
        logger.info("Successfully loaded FAISS index")
        return vectorstore
    except Exception as e:
        logger.error(f"Error loading vectorstore: {e}")
        import traceback

        logger.debug(traceback.format_exc())
        return None


# Load the vectorstore when module is imported
try:
    vectorstore = load_vectorstore()
    if vectorstore is None:
        logger.warning("FAISS vectorstore could not be loaded.")
except Exception as e:
    logger.error(f"Error loading vectorstore: {e}")
    vectorstore = None


def search(query: str, return_type=str, results: int = 5):
    """
    Search for relevant chunks using similarity search.

    Args:
        query: The search query
        return_type: Return as string or list (default: str)
        results: Number of results to return (default: 5)

    Returns:
        Results as string or list depending on return_type
    """
    if vectorstore is None:
        raise ValueError("Vectorstore not loaded. Please ensure FAISS index exists.")

    search_results = vectorstore.similarity_search(query, k=results)

    if return_type == str:
        str_results = ""
        for idx, result in enumerate(search_results, start=1):
            str_results += f"Result {idx}:\n"
            str_results += result.page_content + "\n"
            str_results += "------\n"
        return str_results
    elif return_type == list:
        return [result.page_content for result in search_results]
    else:
        raise ValueError("Invalid return_type. Use str or list.")


# Load questions from saved data
def load_qa_data(questions_path=None):
    """
    Load the pre-generated questions

    Args:
        questions_path: Path to questions file (default: PROCESSED_DATA_DIR / "questions.jsonl")

    Returns:
        List of question-answer pairs
    """
    try:
        if questions_path is None:
            questions_path = PROCESSED_DATA_DIR / "questions.jsonl"

        logger.info(f"Loading questions from: {questions_path}")

        # Load the questions
        with open(questions_path, "r") as f:
            questions = [json.loads(line) for line in f]

        logger.info(f"Successfully loaded {len(questions)} questions")
        return questions
    except Exception as e:
        logger.error(f"Error loading QA data: {e}")
        import traceback

        logger.debug(traceback.format_exc())
        return None


# Load questions when module is imported
try:
    questions = load_qa_data()
    if questions is None:
        logger.warning("Could not load QA data.")
except Exception as e:
    logger.error(f"Error initializing QA data: {e}")
    questions = None


def get_question_answer(idx=None, return_both: bool = True) -> dict:
    """
    Get a question-answer pair either by index or randomly.

    Args:
        idx: Index of the question to retrieve (if None, selects random question)
        return_both: Whether to return both question and answer (default: True)

    Returns:
        Question and answer as tuple if return_both=True, otherwise just the question
    """
    if questions is None:
        raise ValueError("Questions not loaded. Please ensure questions.json exists.")

    if idx is None:
        # Select a random question
        qa_pair = random.choice(questions)
    elif 0 <= idx < len(questions):
        # Select question by index
        qa_pair = questions[idx]
    else:
        raise ValueError(f"Index out of range. Must be between 0 and {len(questions) - 1}")

    question = qa_pair["question"]
    answer = qa_pair["answer"]

    if return_both:
        return {"question": question, "answer": answer}
    else:
        return question


# Function to get the total number of questions
def get_question_count() -> int:
    """Get the total number of available questions"""
    if questions is None:
        raise ValueError("Questions not loaded. Please ensure questions.json exists.")
    return len(questions)


def get_qa_dataset(randomize: bool = False, test_size: float = 0.1, seed: int = 42, questions_path=None) -> tuple:
    """
    Return a HuggingFace Dataset containing question and answer pairs.

    This dataset is constructed from the loaded questions data.
    Each element in the dataset is a dictionary that includes at least:
      - "question": The question text.
      - "answer": The corresponding answer text.
      - "supporting_paragraphs": The supporting paragraphs for the question.
    Additional keys present in the original questions data will also be included.

    Args:
        randomize: Whether to shuffle the dataset
        test_size: Proportion of the dataset to include in the test split (0 for train-only)
        seed: Random seed for reproducibility
        questions_path: Path to questions.jsonl file (if None, uses globally loaded questions)

    Returns:
        A tuple of (train_dataset, test_dataset) HuggingFace Dataset objects.
        If test_size=0, test_dataset will be empty. If test_size=1, train_dataset will be empty.
    """
    qa_data = questions

    if questions_path is not None:
        qa_data = load_qa_data(questions_path)

    if qa_data is None:
        raise ValueError("Questions not loaded. Please ensure questions.jsonl exists.")

    qa_dataset = Dataset.from_list(qa_data)
    if randomize:
        qa_dataset = qa_dataset.shuffle(seed=seed)

    # Create empty dataset for when train or test size is 0
    empty_dataset = Dataset.from_list([])

    if test_size <= 0:
        # Only train dataset, empty test dataset
        train_dataset = qa_dataset
        train_dataset = train_dataset.rename_column("question", "prompt")
        return train_dataset, empty_dataset
    elif test_size >= 1:
        # Only test dataset, empty train dataset
        test_dataset = qa_dataset
        test_dataset = test_dataset.rename_column("question", "prompt")
        return empty_dataset, test_dataset
    else:
        # Both train and test datasets
        split = qa_dataset.train_test_split(test_size=test_size, seed=seed)
        train_dataset = split["train"]
        test_dataset = split["test"]

        # rename the column of the dataset from "question" to "input"
        train_dataset = train_dataset.rename_column("question", "prompt")
        test_dataset = test_dataset.rename_column("question", "prompt")
        return train_dataset, test_dataset
