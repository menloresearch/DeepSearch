"""
Reward functions for RL training.
"""

import json
import re
from datetime import datetime
from pathlib import Path

import numpy as np

from src.config import LOG_FOLDER, logger
from src.evaluation import check_student_answers


def build_reward_correctness_fn(
    vllm_generate_func,
    tokenizer,
):
    """Build a reward function that checks correctness of student answers.

    Args:
        vllm_generate_func: Function to generate answers using vLLM
        tokenizer: Tokenizer for the model

    Returns:
        A reward function that takes prompts and completions and returns correctness scores
    """

    def reward_correctness(prompts: list, completions: list, **reward_kwargs) -> list:
        """Calculate reward based on correctness of student answers.

        Args:
            prompts: List of input prompts
            completions: List of model completions
            **reward_kwargs: Additional arguments for reward calculation

        Returns:
            List of correctness scores between 0 and 1
        """
        # correctness: must be assistant response and must have only one pair of answer tags
        # should not have search tags and information tags
        teacher_answers = reward_kwargs["answer"]
        student_final_messages = [completion["messages"][-1]["content"] for completion in completions]
        student_final_message_roles = [completion["messages"][-1]["role"] for completion in completions]
        is_assistant_response = [role == "assistant" for role in student_final_message_roles]
        has_answer_tag = [re.search(r"<answer>[\s\S]*?</answer>", ans) is not None for ans in student_final_messages]
        has_search_tag = [re.search(r"<search>[\s\S]*?</search>", ans) is not None for ans in student_final_messages]
        has_information_tag = [
            re.search(r"<information>[\s\S]*?</information>", ans) is not None for ans in student_final_messages
        ]

        might_be_correct = check_student_answers(
            prompts,
            teacher_answers,
            student_final_messages,
            vllm_generate_func=vllm_generate_func,
            tokenizer=tokenizer,
        )

        # Convert lists to numpy arrays for element-wise operations
        might_be_correct = np.array(might_be_correct)
        is_assistant_response = np.array(is_assistant_response)
        has_answer_tag = np.array(has_answer_tag)
        has_search_tag = np.array(has_search_tag)
        has_information_tag = np.array(has_information_tag)

        # might be correct and is assistant response and has answer tag and no search or information tags
        correct = might_be_correct & is_assistant_response & has_answer_tag & ~has_search_tag & ~has_information_tag

        # Convert numpy array back to list for return
        correct = correct.tolist()

        # Log correctness metrics with length info
        logger.info(f"Correctness metrics: {correct}")
        logger.info(f"Average correctness: {np.mean(correct):.2f}")
        logger.info(f"Standard deviation: {np.std(correct):.2f}")

        # Log length metrics
        student_lengths = [len(ans.strip()) for ans in student_final_messages]
        teacher_lengths = [len(ans.strip()) for ans in teacher_answers]
        logger.info(f"Student lengths: {student_lengths}")
        logger.info(f"Teacher lengths: {teacher_lengths}")
        logger.info(f"Average student length: {np.mean(student_lengths):.2f}")
        logger.info(f"Average teacher length: {np.mean(teacher_lengths):.2f}")
        logger.info(f"Length ratio: {np.mean(student_lengths) / np.mean(teacher_lengths):.2f}")

        # Log chat state
        log_chat_state(
            prompts=prompts,
            completions=completions,
            rewards=correct,
            reward_type="correctness",
            teacher_answers=teacher_answers,
            validation_results={
                "is_assistant": is_assistant_response,
                "has_answer": has_answer_tag,
                "has_search": has_search_tag,
                "has_info": has_information_tag,
                "might_be_correct": might_be_correct,
            },
        )

        return correct

    return reward_correctness


def reward_format(prompts: list, completions: list, **reward_kwargs) -> list:
    """Reward function that checks if the completion follows the required format with proper tags.

    Args:
        prompts: List of input prompts
        completions: List of completion dictionaries containing messages
        **reward_kwargs: Additional reward parameters

    Returns:
        list: List of rewards (1.0 for valid format, 0.0 for invalid)
    """
    # Regex patterns for each tag type - no markdown allowed
    think_pattern = r"<think>[\s\S]*?</think>"
    search_pattern = r"<search>[\s\S]*?</search>"
    answer_pattern = r"<answer>[\s\S]*?</answer>"

    # Information tag patterns - handle multiple variants
    info_patterns = [
        r"<information>[\s\S]*?</information>",  # Standard
        r"<info>[\s\S]*?</info>",  # Shortened
        r"<Info[\w]*>[\s\S]*?</Info[\w]*>",  # Capitalized variants
        r"<INFORMATION>[\s\S]*?</INFORMATION>",  # Uppercase
        r"<INFO>[\s\S]*?</INFO>",  # Uppercase shortened
    ]

    # Invalid patterns (bold/italic tags)
    invalid_patterns = [
        r"\*\*<\/?(?:think|search|answer|information|info)>\*\*",  # Bold tags
        r"\*<\/?(?:think|search|answer|information|info)>\*",  # Italic tags
        r"_<\/?(?:think|search|answer|information|info)>_",  # Underscore italic
    ]

    rewards = []
    validation_results = {
        "has_think": [],
        "has_answer": [],
        "has_search": [],
        "has_invalid_tags": [],
        "has_info_tags": [],
    }

    for completion in completions:
        messages = completion.get("messages", [])
        assistant_msgs = [msg["content"] for msg in messages if msg["role"] == "assistant"]

        if not assistant_msgs:
            rewards.append(0.0)
            for key in validation_results:
                validation_results[key].append(False)
            continue

        content = assistant_msgs[-1]

        has_invalid_tags = any(re.search(pattern, content) for pattern in invalid_patterns)
        validation_results["has_invalid_tags"].append(has_invalid_tags)
        if has_invalid_tags:
            rewards.append(0.0)
            for key in ["has_think", "has_answer", "has_search", "has_info_tags"]:
                validation_results[key].append(False)
            continue

        has_info_tags = False
        for pattern in info_patterns:
            if re.findall(pattern, content, re.IGNORECASE):
                has_info_tags = True
                break
        validation_results["has_info_tags"].append(has_info_tags)

        if has_info_tags:
            rewards.append(0.0)
            for key in ["has_think", "has_answer", "has_search"]:
                validation_results[key].append(False)
            continue

        think_matches = re.findall(think_pattern, content)
        search_matches = re.findall(search_pattern, content)
        answer_matches = re.findall(answer_pattern, content)

        has_think = len(think_matches) >= 1
        has_answer = len(answer_matches) == 1
        has_search = len(search_matches) >= 1

        validation_results["has_think"].append(has_think)
        validation_results["has_answer"].append(has_answer)
        validation_results["has_search"].append(has_search)

        if has_search and has_answer:
            rewards.append(0.0)
            continue

        reward = 1.0 if has_think and (has_answer or has_search) else 0.0
        rewards.append(reward)

        if not reward:
            logger.debug(f"Format issues - think: {has_think}, answer: {has_answer}, search: {has_search}")
            if search_matches:
                logger.debug(f"Number of search tags: {len(search_matches)}")

    logger.info(f"Format reward metrics - Mean: {np.mean(rewards):.3f}, Valid formats: {sum(rewards)}/{len(rewards)}")

    # Log chat state with validation results
    log_chat_state(
        prompts=prompts,
        completions=completions,
        rewards=rewards,
        reward_type="format",
        validation_results=validation_results,
    )

    return rewards


# TODO: Implement this reward function if the project survives
def reward_long_query(completions, **kwargs):
    """Reward function that checks if the query is long."""
    pass


def reward_retry(prompts: list, completions: list, **reward_kwargs) -> list:
    """
    Reward function that encourages optimal retry behavior.
    Rewards increase with more search attempts but caps at optimal_search_count.
    Penalizes having multiple searches in a single message.

    Args:
        prompts: List of input prompts
        completions: List of completion dictionaries with messages
        **reward_kwargs: Additional reward parameters (chunk_id, answer, etc.)

    Returns:
        List of rewards for each completion, rounded to 3 decimal places
    """
    rewards = []
    search_queries = []
    violations = []

    # Config for retry rewards
    optimal_search_count = 5  # Cap rewards at this many searches
    base_reward = 0.2  # Base reward for having at least one search
    increment = 0.15  # Reward increment per search attempt (0.2 + 5*0.15 = 0.95 max)
    violation_penalty = 0.5  # Penalty for having multiple searches in one message

    # Regex pattern for search tags
    search_pattern = r"<search>[\s\S]*?</search>"

    for completion in completions:
        # Get assistant messages
        assistant_messages = [msg["content"] for msg in completion["messages"] if msg["role"] == "assistant"]

        # Count search tags in assistant messages
        message_searches = []
        for msg in assistant_messages:
            # Find all search tags in each message
            search_matches = re.findall(search_pattern, msg)
            message_searches.append(len(search_matches))

        # Record total search queries
        total_searches = sum(message_searches)
        search_queries.append(total_searches)

        # Check for violations (more than one search query per message)
        violation = any(count > 1 for count in message_searches)
        violations.append(violation)

        # Calculate reward
        if total_searches == 0:
            reward = 0.0  # No searches = no reward
        else:
            # Base reward for having at least one search
            reward = base_reward

            # Add incremental reward for each search up to optimal_search_count
            search_bonus = min(total_searches, optimal_search_count) * increment
            reward += search_bonus

            # Cap reward at 1.0
            reward = min(1.0, reward)

            # Apply penalty if there's a violation
            if violation:
                reward *= 1 - violation_penalty

            # Round to 3 decimal places to avoid floating point precision issues
            reward = round(reward, 3)

        rewards.append(reward)

    # Log metrics with search distribution info
    logger.info(f"Retry behavior rewards: {np.mean(rewards):.3f} ± {np.std(rewards):.3f}")
    logger.info(f"Search tags per completion: {np.mean(search_queries):.2f} ± {np.std(search_queries):.2f}")
    logger.info(f"Violations (>1 search per message): {sum(violations)}/{len(violations)}")
    logger.info(f"Search counts distribution: {search_queries}")

    # Log chat state
    log_chat_state(
        prompts=prompts,
        completions=completions,
        rewards=rewards,
        reward_type="retry",
        search_counts=search_queries,
        violations=violations,
        optimal_search_count=optimal_search_count,
    )

    return rewards


def reward_em_chunk(prompts: list, completions: list, **reward_kwargs) -> list:
    """Reward function that checks if model's search queries hit the correct chunk content.

    Args:
        prompts: List of input prompts
        completions: List of completion dictionaries with messages
        **reward_kwargs: Additional reward parameters including:
            - chunk_content: List of correct chunk contents to match against
            - step: Optional step number for logging metrics

    Returns:
        list: List of rewards (1.0 for exact match, 0.0 otherwise)

    Raises:
        ValueError: If chunk_content is not provided in reward_kwargs
    """
    logger.debug(f"Calculating rewards for {len(prompts)} prompts")

    # Get correct chunk contents from reward kwargs
    correct_contents = reward_kwargs.get("chunk_content", [])
    if not correct_contents:
        logger.error("No chunk_content provided in reward_kwargs")
        raise ValueError("chunk_content must be provided in reward_kwargs")

    rewards = []
    for i, (completion, correct_content) in enumerate(zip(completions, correct_contents)):
        # Get all messages from ipython or user roles that start with <information>
        search_results = [
            msg["content"]
            for msg in completion["messages"]
            if msg["role"] in ("ipython", "user") and msg["content"].strip().startswith("<information>")
        ]
        logger.debug(f"Found {len(search_results)} search results for prompt {i}")

        # Log ground truth and searched chunks for debugging
        logger.info(f"📝 Ground Truth Chunk: {correct_content}")
        for j, result in enumerate(search_results):
            logger.info(f"🔍 Searched Chunk {j + 1}: {result}")

        # Check if any search hit the correct chunk content
        found_correct_chunk = any(correct_content in result for result in search_results)

        if not found_correct_chunk:
            logger.warning(
                f"Failed to find correct chunk for prompt {i}:\n"
                f"Search results: {[r[:100] + '...' for r in search_results]}"
            )

        reward = 1.0 if found_correct_chunk else 0.0
        rewards.append(reward)
        logger.debug(f"Reward for prompt {i}: {reward}")

    # Log summary metrics
    logger.info("Chunk Query Rewards Summary:")
    logger.info(f"Total prompts: {len(prompts)}")
    logger.info(f"Correct matches: {sum(rewards)}")
    logger.info(f"Average reward: {np.mean(rewards):.3f}")
    logger.info(f"Reward std: {np.std(rewards):.3f}")

    # Log chat state
    log_chat_state(
        prompts=prompts,
        completions=completions,
        rewards=rewards,
        reward_type="em_chunk",
        correct_contents=correct_contents,
    )

    return rewards


def log_chat_state(prompts: list, completions: list, rewards: list, reward_type: str, **kwargs) -> None:
    """Log chat state and rewards to JSONL file.

    Args:
        prompts: List of input prompts
        completions: List of model completions
        rewards: List of calculated rewards
        reward_type: Type of reward function used
        **kwargs: Additional data to log
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    chat_states_dir = LOG_FOLDER / "chat_states"
    chat_states_dir.mkdir(parents=True, exist_ok=True)

    # Convert numpy arrays to lists in kwargs
    for key, value in kwargs.items():
        if isinstance(value, dict):
            for k, v in value.items():
                if isinstance(v, np.ndarray):
                    kwargs[key][k] = v.tolist()
        elif isinstance(value, np.ndarray):
            kwargs[key] = value.tolist()

    # Create one JSONL file per reward type
    log_file = chat_states_dir / f"chat_states_{reward_type}.jsonl"

    # Append each chat state as a new line
    with open(log_file, "a", encoding="utf-8") as f:
        for prompt, completion, reward in zip(prompts, completions, rewards):
            chat_state = {
                "timestamp": timestamp,
                "reward_type": reward_type,
                "prompt": prompt,
                "messages": completion["messages"],
                "reward": float(reward) if isinstance(reward, (np.number, np.ndarray)) else reward,
                "metadata": kwargs,
            }
            f.write(json.dumps(chat_state, ensure_ascii=False) + "\n")

    logger.info(f"💾 Appended {len(prompts)} chat states to {log_file}")
