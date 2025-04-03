"""
Reward functions for RL training.
"""

import re

import numpy as np

from src.config import logger
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
        teacher_answers = reward_kwargs["answer"]
        student_answers = [completion["messages"][-1]["content"] for completion in completions]

        # Log non-exact matches
        for i, (student, teacher) in enumerate(zip(student_answers, teacher_answers)):
            if student.strip().lower() != teacher.strip().lower():
                logger.debug(f"Non-exact match at index {i}:\nStudent: {student}\nTeacher: {teacher}")

        correct = check_student_answers(
            prompts,
            teacher_answers,
            student_answers,
            vllm_generate_func=vllm_generate_func,
            tokenizer=tokenizer,
        )

        # Log correctness metrics with length info
        logger.info(f"Correctness metrics: {correct}")
        logger.info(f"Average correctness: {np.mean(correct):.2f}")
        logger.info(f"Standard deviation: {np.std(correct):.2f}")

        # Log length metrics
        student_lengths = [len(ans.strip()) for ans in student_answers]
        teacher_lengths = [len(ans.strip()) for ans in teacher_answers]
        logger.info(f"Student lengths: {student_lengths}")
        logger.info(f"Teacher lengths: {teacher_lengths}")
        logger.info(f"Average student length: {np.mean(student_lengths):.2f}")
        logger.info(f"Average teacher length: {np.mean(teacher_lengths):.2f}")
        logger.info(f"Length ratio: {np.mean(student_lengths) / np.mean(teacher_lengths):.2f}")

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

    for completion in completions:
        messages = completion.get("messages", [])
        assistant_msgs = [msg["content"] for msg in messages if msg["role"] == "assistant"]

        if not assistant_msgs:
            rewards.append(0.0)
            continue

        content = assistant_msgs[-1]  # Get the last assistant message

        # Check for invalid markdown formatting
        has_invalid_tags = any(re.search(pattern, content) for pattern in invalid_patterns)
        if has_invalid_tags:
            logger.debug("Found markdown-formatted tags in response")
            rewards.append(0.0)
            continue

        # Check for any information tag variants (should not exist in assistant messages)
        has_info_tags = False
        for pattern in info_patterns:
            info_matches = re.findall(pattern, content, re.IGNORECASE)
            if info_matches:
                logger.debug(f"Found {len(info_matches)} information tag(s) of type '{pattern}' in assistant message")
                has_info_tags = True
                break

        if has_info_tags:
            rewards.append(0.0)
            continue

        # Find all tag matches
        think_matches = re.findall(think_pattern, content)
        search_matches = re.findall(search_pattern, content)
        answer_matches = re.findall(answer_pattern, content)

        # Verify tag presence and count
        has_think = len(think_matches) >= 1
        has_answer = len(answer_matches) == 1  # Must have exactly one answer
        has_search = len(search_matches) >= 1  # One or more search tags

        # Check for search and answer in the same message (not allowed)
        if has_search and has_answer:
            logger.debug("Found both search and answer tags in the same message")
            rewards.append(0.0)
            continue

        # Award reward - must have think tag and either answer or search (but not both)
        reward = 1.0 if has_think and (has_answer or has_search) else 0.0
        rewards.append(reward)

        # Log issues for debugging
        if not reward:
            logger.debug(f"Format issues - think: {has_think}, answer: {has_answer}, search: {has_search}")
            if search_matches:
                logger.debug(f"Number of search tags: {len(search_matches)}")

    # Log overall metrics
    logger.info(f"Format reward metrics - Mean: {np.mean(rewards):.3f}, Valid formats: {sum(rewards)}/{len(rewards)}")

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

    return rewards
