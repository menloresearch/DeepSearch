"""
Reward functions for RL training.
"""

import json
import re
from datetime import datetime
from difflib import SequenceMatcher

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
        "ends_properly": [],  # New validation result
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

        # Check if content ends with </search> or </answer> (ignoring whitespace)
        content_stripped = content.strip()
        ends_properly = content_stripped.endswith("</search>") or content_stripped.endswith("</answer>")
        validation_results["ends_properly"].append(ends_properly)

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

        # Check for proper tag sequence - think must come before answer/search
        if has_answer or has_search:
            last_think_pos = content.rfind("</think>")
            answer_pos = content.find("<answer>") if has_answer else float("inf")
            search_pos = content.find("<search>") if has_search else float("inf")
            tag_pos = min(answer_pos, search_pos)

            if last_think_pos == -1 or last_think_pos > tag_pos:
                rewards.append(0.0)
                continue

        # Only reward if format is valid AND response ends properly
        reward = 1.0 if has_think and (has_answer or has_search) and ends_properly else 0.0
        rewards.append(reward)

        if not reward:
            logger.debug(
                f"Format issues - think: {has_think}, answer: {has_answer}, search: {has_search}, ends_properly: {ends_properly}"
            )
            if search_matches:
                logger.debug(f"Number of search tags: {len(search_matches)}")

    logger.info(f"Format reward metrics - Mean: {np.mean(rewards):.3f}, Valid formats: {sum(rewards)}/{len(rewards)}")
    logger.info(f"Responses ending properly: {sum(validation_results['ends_properly'])}/{len(rewards)}")

    # Log chat state with validation results
    log_chat_state(
        prompts=prompts,
        completions=completions,
        rewards=rewards,
        reward_type="format",
        validation_results=validation_results,
    )

    return rewards


def reward_retry(prompts: list, completions: list, **reward_kwargs) -> list:
    """
    Reward function that encourages optimal retry behavior.
    Rewards increase with more search attempts but caps at optimal_search_count.
    Penalizes having multiple searches in a single message.
    Returns 0 if final message doesn't contain answer tags.

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

    # Regex pattern for search and answer tags
    search_pattern = r"<search>[\s\S]*?</search>"
    answer_pattern = r"<answer>[\s\S]*?</answer>"

    for completion in completions:
        # Get assistant messages
        assistant_messages = [msg["content"] for msg in completion["messages"] if msg["role"] == "assistant"]

        if not assistant_messages:
            rewards.append(0.0)
            search_queries.append(0)
            violations.append(False)
            continue

        # Check if final message contains answer tags
        final_message = assistant_messages[-1]
        has_answer = bool(re.search(answer_pattern, final_message))

        if not has_answer:
            rewards.append(0.0)
            search_queries.append(0)
            violations.append(False)
            continue

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


def tag_count_reward(prompts: list, completions: list, **reward_kwargs) -> list:
    """Reward function that checks for proper tag counts in the conversation.

    Rewards:
    - 0.1 for each proper pair of think tags in each assistant message
    - 0.5 for having exactly one pair of answer tags in entire conversation
    - 0.1 for each proper pair of search tags

    Args:
        prompts: List of input prompts
        completions: List of completion dictionaries with messages
        **reward_kwargs: Additional reward parameters

    Returns:
        list: List of rewards between 0 and 1
    """
    rewards = []
    validation_results = {
        "think_pairs_per_msg": [],  # List of lists, each inner list has think pair counts per assistant msg
        "answer_pairs": [],  # Total answer pairs in conversation
        "search_pairs": [],  # Total search pairs in conversation
    }

    for completion in completions:
        # Get all assistant messages
        assistant_msgs = [msg["content"] for msg in completion["messages"] if msg["role"] == "assistant"]

        if not assistant_msgs:
            rewards.append(0.0)
            validation_results["think_pairs_per_msg"].append([])
            validation_results["answer_pairs"].append(0)
            validation_results["search_pairs"].append(0)
            continue

        # Count think pairs per assistant message
        think_pairs_per_msg = []
        for msg in assistant_msgs:
            # Count complete think tag pairs
            think_opens = len(re.findall(r"<think>", msg))
            think_closes = len(re.findall(r"</think>", msg))
            think_pairs = min(think_opens, think_closes)
            think_pairs_per_msg.append(think_pairs)

        # Count answer tags in entire conversation (should be exactly one pair)
        total_answer_opens = sum(msg.count("<answer>") for msg in assistant_msgs)
        total_answer_closes = sum(msg.count("</answer>") for msg in assistant_msgs)
        answer_pairs = min(total_answer_opens, total_answer_closes)

        # Count search tags
        total_search_opens = sum(msg.count("<search>") for msg in assistant_msgs)
        total_search_closes = sum(msg.count("</search>") for msg in assistant_msgs)
        search_pairs = min(total_search_opens, total_search_closes)

        # Calculate reward components
        think_reward = sum(min(pairs, 1) * 0.1 for pairs in think_pairs_per_msg)  # 0.1 per msg with proper think pair
        answer_reward = 0.5 if answer_pairs == 1 else 0.0  # 0.5 for exactly one answer pair
        search_reward = min(search_pairs, 1) * 0.1  # 0.1 for having search pairs

        total_reward = min(think_reward + answer_reward + search_reward, 1.0)
        rewards.append(total_reward)

        # Store validation results
        validation_results["think_pairs_per_msg"].append(think_pairs_per_msg)
        validation_results["answer_pairs"].append(answer_pairs)
        validation_results["search_pairs"].append(search_pairs)

        # Debug logging
        if total_reward < 1.0:
            logger.debug(
                f"Tag count issues - think_pairs: {think_pairs_per_msg}, "
                f"answer_pairs: {answer_pairs}, search_pairs: {search_pairs}"
            )

    # Log metrics
    logger.info(
        f"Tag count reward metrics - Mean: {np.mean(rewards):.3f}, Perfect scores: {sum(r == 1.0 for r in rewards)}/{len(rewards)}"
    )
    logger.info(
        f"Average think pairs per message: {np.mean([np.mean(pairs) if pairs else 0 for pairs in validation_results['think_pairs_per_msg']]):.2f}"
    )
    logger.info(
        f"Conversations with exactly one answer pair: {sum(pairs == 1 for pairs in validation_results['answer_pairs'])}/{len(rewards)}"
    )

    # Log chat state
    log_chat_state(
        prompts=prompts,
        completions=completions,
        rewards=rewards,
        reward_type="tag_count",
        validation_results=validation_results,
    )

    return rewards


def reward_search_strategy(prompts: list, completions: list, **reward_kwargs) -> list:
    """Reward function that checks for good search strategy and query analysis steps.

    The expected conversation flow pattern is:
    1. Initial search: question -> assistant(think + search)
    2. Process info: information -> assistant(think + refined search)
    3. Final answer: information -> assistant(think + answer)

    Rewards:
    - Initial search (0.2): Starting with broad/overview search
    - Information processing (0.4): Analyzing provided info and refining search
    - Final synthesis (0.4): Analyzing all info and providing final answer

    Args:
        prompts: List of input prompts
        completions: List of completion dictionaries
        **reward_kwargs: Additional reward parameters

    Returns:
        list: List of rewards between 0 and 1
    """
    rewards = []
    validation_results = {
        "initial_search": [],  # First search attempt
        "info_processing": [],  # Number of info-based refinements
        "final_synthesis": [],  # Final answer with proper analysis
    }

    # Patterns for conversation flow
    think_pattern = r"<think>[^<>]+</think>"
    search_pattern = r"<search>[^<>]+</search>"
    answer_pattern = r"<answer>[^<>]+</answer>"
    info_pattern = r"<information>[^<>]+</information>"

    # Analysis patterns
    info_analysis_pattern = (
        r"<think>[^<>]*?\b(?:based|according|from|results?|found|shows?|provided|information)\b[^<>]*?</think>"
    )

    for completion in completions:
        messages = completion.get("messages", [])
        if not messages:
            rewards.append(0.0)
            for key in validation_results:
                validation_results[key].append(False)
            continue

        # Track conversation flow
        has_initial_search = False
        info_based_searches = 0
        has_final_synthesis = False

        # Track current state
        last_was_info = False
        search_after_info = 0
        analysis_after_info = 0

        for i, msg in enumerate(messages):
            content = msg["content"]
            role = msg["role"]

            if role == "assistant":
                has_think = bool(re.search(think_pattern, content))
                has_search = bool(re.search(search_pattern, content))
                has_answer = bool(re.search(answer_pattern, content))
                has_info_analysis = bool(re.search(info_analysis_pattern, content, re.IGNORECASE))

                # Check initial search (first assistant message with search)
                if not has_initial_search and has_think and has_search:
                    has_initial_search = True

                # Check info-based refinement
                if last_was_info and has_think:
                    if has_search:
                        search_after_info += 1
                    if has_info_analysis:
                        analysis_after_info += 1

                # Check final synthesis
                if has_answer and has_think and has_info_analysis:
                    has_final_synthesis = True

            elif role in ["user", "ipython"] and re.search(info_pattern, content):
                last_was_info = True
            else:
                last_was_info = False

        # Calculate rewards
        initial_reward = 0.2 if has_initial_search else 0.0

        # Info processing reward: proper analysis and search after info
        info_processing = min(search_after_info, analysis_after_info)  # Must have both analysis and search
        info_reward = min(0.4, 0.2 * info_processing)  # 0.2 per proper info-based refinement, max 0.4

        # Final synthesis reward
        synthesis_reward = 0.4 if has_final_synthesis else 0.0

        total_reward = initial_reward + info_reward + synthesis_reward
        rewards.append(total_reward)

        # Store validation results
        validation_results["initial_search"].append(has_initial_search)
        validation_results["info_processing"].append(info_processing)
        validation_results["final_synthesis"].append(has_final_synthesis)

        # Debug logging
        if total_reward < 0.6:  # Log if missing significant components
            logger.debug(
                f"Search flow issues - initial: {has_initial_search}, "
                f"info_processing: {info_processing}, "
                f"final_synthesis: {has_final_synthesis}"
            )

    # Log metrics
    logger.info(
        f"Search strategy metrics - Mean: {np.mean(rewards):.3f}, Perfect scores: {sum(r == 1.0 for r in rewards)}/{len(rewards)}"
    )
    logger.info(f"Initial searches: {sum(validation_results['initial_search'])}/{len(rewards)}")
    logger.info(f"Average info processing steps: {np.mean([r for r in validation_results['info_processing']]):.2f}")
    logger.info(f"Final synthesis rate: {sum(validation_results['final_synthesis'])}/{len(rewards)}")

    # Log chat state
    log_chat_state(
        prompts=prompts,
        completions=completions,
        rewards=rewards,
        reward_type="search_strategy",
        validation_results=validation_results,
    )

    return rewards


def reward_search_diversity(prompts: list, completions: list, **reward_kwargs) -> list:
    """Reward function that evaluates diversity of search queries in a conversation.

    Rewards higher diversity in search queries and penalizes repetitive searches.
    Uses string similarity to compare queries, with diminishing returns for
    similar queries.

    Scoring:
    - Base reward: 0.2 per unique query concept (max 0.4)
    - Diversity bonus: Up to 0.4 based on semantic diversity
    - Operator bonus: Up to 0.2 for proper use of search operators
    - Penalties:
      * Similar queries (>0.8 similarity): -0.1 per pair
      * Exact duplicates: -0.2 per duplicate

    Args:
        prompts: List of input prompts
        completions: List of completion dictionaries
        **reward_kwargs: Additional reward parameters

    Returns:
        list: List of rewards between 0 and 1
    """

    def normalize_query(query: str) -> tuple[str, list[str]]:
        """Normalize search query for comparison."""
        # Extract operators before normalization
        operators = re.findall(r'(?:site|filetype):\S+|"[^"]+"|(?:\s+OR\s+|\s+AND\s+|-\w+)', query)
        # Remove operators for base comparison
        base_query = re.sub(r'(?:site|filetype):\S+|"[^"]+"|(?:\s+OR\s+|\s+AND\s+|-\w+)', "", query.lower())
        # Remove special chars and extra spaces from base query
        base_query = re.sub(r"[^\w\s]", " ", base_query)
        return " ".join(base_query.split()), operators

    def query_similarity(q1: str, q2: str) -> float:
        """Calculate similarity between two queries."""
        # Compare normalized base queries
        base1, ops1 = normalize_query(q1)
        base2, ops2 = normalize_query(q2)

        # Base similarity from query text
        base_sim = SequenceMatcher(None, base1, base2).ratio()

        # Significantly reduce similarity if using different operators
        if ops1 != ops2:
            # More operators = more different
            unique_ops = len(set(ops1) ^ set(ops2))  # XOR to get unique operators
            base_sim *= max(0.3, 1.0 - (unique_ops * 0.2))  # Each unique operator reduces similarity by 20%

        return base_sim

    rewards = []

    for completion in completions:
        # Extract all search queries from assistant messages
        search_queries = []
        for msg in completion.get("messages", []):
            if msg["role"] == "assistant":
                # Find all search tags
                searches = re.findall(r"<search>([^<>]+)</search>", msg["content"])
                search_queries.extend(searches)

        if not search_queries:
            rewards.append(0.0)
            continue

        # Calculate diversity score
        total_queries = len(search_queries)
        if total_queries == 1:
            rewards.append(0.2)  # Base reward for single query
            continue

        # Calculate pairwise similarities and track duplicates/high similarities
        similarity_sum = 0
        pair_count = 0
        similar_pairs = 0  # Count pairs with >0.8 similarity
        exact_duplicates = 0  # Count exact matches

        # Count unique operators and track their usage
        all_operators = set()
        operator_usage = []  # Track operators per query
        for query in search_queries:
            _, ops = normalize_query(query)
            all_operators.update(ops)
            operator_usage.append(len(ops))

        # Track normalized queries to find duplicates
        seen_queries = set()
        unique_queries = []

        for i in range(total_queries):
            base_i, _ = normalize_query(search_queries[i])
            if base_i in seen_queries:
                exact_duplicates += 1
            else:
                unique_queries.append(search_queries[i])
            seen_queries.add(base_i)

            for j in range(i + 1, total_queries):
                similarity = query_similarity(search_queries[i], search_queries[j])
                similarity_sum += similarity
                pair_count += 1

                # Count highly similar pairs (ignoring operator differences)
                base_i, _ = normalize_query(search_queries[i])
                base_j, _ = normalize_query(search_queries[j])
                base_sim = SequenceMatcher(None, base_i, base_j).ratio()
                if base_sim > 0.8 and base_sim < 1.0:  # Don't count exact duplicates twice
                    similar_pairs += 1

        # Average similarity (0-1), weighted less for operator differences
        avg_similarity = similarity_sum / pair_count if pair_count > 0 else 0

        # Calculate diversity score (1 - avg_similarity)
        diversity_score = 1 - avg_similarity

        # Calculate operator bonus (up to 0.2)
        # Reward both variety and consistent usage
        operator_variety_bonus = min(0.15, len(all_operators) * 0.05)  # Up to 0.15 for unique operators
        operator_usage_ratio = sum(1 for x in operator_usage if x > 0) / total_queries
        operator_usage_bonus = 0.05 * operator_usage_ratio  # Up to 0.05 for consistent usage
        operator_bonus = operator_variety_bonus + operator_usage_bonus

        # Calculate penalties
        # Reduce penalties when operators are different
        similarity_penalty = similar_pairs * 0.1  # Reduced penalty for similar pairs
        if len(all_operators) >= 2:  # If using multiple operators, reduce penalties
            similarity_penalty *= 0.5

        duplicate_penalty = exact_duplicates * 0.2  # Keep strong penalty for exact duplicates

        # Final reward calculation:
        # - Base reward per unique query (max 0.4)
        # - Diversity bonus (up to 0.4)
        # - Operator bonus (up to 0.2)
        # - Apply penalties
        unique_query_count = len(unique_queries)
        base_reward = min(0.4, 0.2 * unique_query_count)
        diversity_bonus = diversity_score * 0.4
        total_reward = base_reward + diversity_bonus + operator_bonus - similarity_penalty - duplicate_penalty

        # Cap at 1.0 and floor at 0.0
        reward = max(0.0, min(1.0, total_reward))

        # Debug logging
        logger.debug(
            f"Search diversity metrics - "
            f"Queries: {total_queries}, "
            f"Unique: {len(seen_queries)}, "
            f"Similar pairs: {similar_pairs}, "
            f"Duplicates: {exact_duplicates}, "
            f"Avg similarity: {avg_similarity:.2f}, "
            f"Diversity score: {diversity_score:.2f}, "
            f"Operator bonus: {operator_bonus:.2f}, "
            f"Penalties: -{similarity_penalty + duplicate_penalty:.2f}, "
            f"Final reward: {reward:.2f}"
        )

        rewards.append(reward)

    # Log overall metrics
    if rewards:
        logger.info(f"Search diversity metrics - Mean reward: {np.mean(rewards):.3f}, Max reward: {max(rewards):.3f}")

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
