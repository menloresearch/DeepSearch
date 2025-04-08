"""
Main package exports for RL helpers.
"""

from trl.trainer.grpo_trainer import apply_chat_template

from config import logger
from src.agent import Agent, extract_search_query
from src.evaluation import check_student_answers, run_eval, verify
from src.prompts import build_user_prompt, format_search_results, get_system_prompt
from src.rewards import (
    build_reward_correctness_fn,
    reward_em_chunk,
    reward_format,
    reward_retry,
)
from src.search_module import get_qa_dataset, search
from src.tokenizer_adapter import LlamaTokenizerAdapter, R1DistilTokenizerAdapter

__all__ = [
    # Prompts
    "get_system_prompt",
    "build_user_prompt",
    "format_search_results",
    "apply_chat_template",
    # Agent
    "Agent",
    "LlamaTokenizerAdapter",
    "R1DistilTokenizerAdapter",
    "extract_search_query",
    # Rewards
    "build_reward_correctness_fn",
    "reward_format",
    "reward_retry",
    "reward_em_chunk",
    # Evaluation
    "run_eval",
    "check_student_answers",
    "verify",
    # Search
    "get_qa_dataset",
    "search",
    "logger",
]
