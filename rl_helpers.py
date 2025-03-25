"""
RL helpers module for handling tool-based conversations.
This module provides utility functions for handling chat-based tool interactions
and calculating rewards based on the quality of responses.
"""

import asyncio
import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import nest_asyncio
import numpy as np
import torch
from loguru import logger

from search_module import get_qa_dataset, search

# Setup loguru
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)
logger.add(
    log_dir / "rl_helpers_{time}.log",
    rotation="500 MB",
    retention="10 days",
    level="DEBUG",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
)

nest_asyncio.apply()
from typing import Callable, List

from trl.trainer.grpo_trainer import apply_chat_template


# Constants for prompts and tool definitions
def get_system_prompt():
    """Get the system prompt with current date."""
    current_date = datetime.now().strftime("%d %b %Y")
    return f"""Cutting Knowledge Date: December 2023
Today Date: {current_date}

When you receive a tool call response, use the output to format an answer to the original user question.

You are a helpful assistant with tool calling capabilities.
"""


# Tool definition for search corpus
SEARCH_TOOL_DEFINITION = {
    "type": "function",
    "function": {
        "name": "search_corpus",
        "description": "Search over the knowledge corpus with a given query",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The query to search the knowledge corpus with",
                },
            },
            "required": ["query"],
        },
    },
}


def build_user_prompt(q):
    """
    Build a user prompt with the question and search tool definition.

    Args:
        q (str): The question to ask

    Returns:
        str: Formatted user prompt
    """
    user_prompt = f"""You are a research assistant, and you use the search_corpus tool to find answers to questions.
Given a question, answer it using by doing searches using the search_corpus tool.
To use the search_corpus tool, respond with a JSON for a function call with its proper arguments.

You may also reason in any message, think step by step about how to answer the question. Wrap your reasoning in <think> and </think> tags.

{json.dumps(SEARCH_TOOL_DEFINITION, indent=2)}

Question: {q}
"""
    return user_prompt


def get_initial_chat(question):
    """
    Initialize a chat state with the question.

    Args:
        question (str): The question to ask

    Returns:
        dict: Initial chat state with system and user messages
    """
    return {
        "messages": [
            {"role": "system", "content": get_system_prompt()},
            {"role": "user", "content": build_user_prompt(question)},
        ]
    }


def extract_json_objects(text):
    """
    Extracts JSON objects (dictionaries) from a text that may contain multiple JSON objects.

    Args:
        text (str): The input text possibly containing JSON objects.

    Returns:
        list: A list of parsed JSON objects (dictionaries) extracted from the text.
    """
    results = []
    length = len(text)
    i = 0

    while i < length:
        # Look for the start of a JSON object
        if text[i] == "{":
            start = i
            stack = 1
            i += 1
            # Continue until we find the matching closing brace
            while i < length and stack > 0:
                if text[i] == "{":
                    stack += 1
                elif text[i] == "}":
                    stack -= 1
                i += 1
            # Only attempt to decode if the braces are balanced
            if stack == 0:
                candidate = text[start:i]
                try:
                    obj = json.loads(candidate)
                    # Optionally, ensure it's a dictionary if that's what you expect
                    if isinstance(obj, dict):
                        results.append(obj)
                except json.JSONDecodeError:
                    # If it's not valid JSON, skip it.
                    pass
        else:
            i += 1
    return results


def remove_reasoning(text: str) -> str:
    """
    Removes all content between <think> and </think> tags,
    including the tags themselves.

    Parameters:
        text (str): The input text that may contain <think>...</think> tags.

    Returns:
        str: The text with the tags and their content removed.
    """
    # The regex pattern matches from <think> to </think> non-greedily.
    pattern = r"<think>.*?</think>"
    cleaned_text = re.sub(pattern, "", text, flags=re.DOTALL)
    return cleaned_text


def run_agent_generations(generate_fn, tokenizer, chat_states):
    """
    Run generation for chat states requiring assistant responses.
    """
    logger.debug(f"Starting generation for {len(chat_states)} chat states")
    prompts = []
    batch_indices = []
    # Prepare prompts for chat states needing an assistant response.
    for idx, chat_state in enumerate(chat_states):
        if chat_state.get("finished"):
            logger.debug(f"Chat state {idx} already finished, skipping")
            continue

        if chat_state["messages"][-1]["role"] in ["ipython", "user"]:
            prompt = apply_chat_template(chat_state, tokenizer=tokenizer)["text"]
            prompts.append(prompt)
            batch_indices.append(idx)
            logger.debug(f"Added prompt for chat state {idx}")

    if prompts:
        logger.info(f"Generating responses for {len(prompts)} prompts")
        responses = generate_fn(prompts)
        for i, idx in enumerate(batch_indices):
            chat_state = chat_states[idx]
            response = responses[i]
            if hasattr(response, "outputs"):
                full_response = response.outputs[0].text
            else:
                full_response = response
            assistant_response = full_response.split(
                "<|start_header_id|>assistant<|end_header_id|>"
            )[-1]
            chat_state["messages"].append(
                {"role": "assistant", "content": assistant_response}
            )
            logger.debug(f"Added assistant response to chat state {idx}")
    else:
        logger.debug("No prompts to generate responses for")
    return chat_states


def check_finished_chats(chat_states):
    """
    Check which chat states are finished (no more function calls).

    Args:
        chat_states: List of chat states

    Returns:
        list: Updated chat states with finished flag
    """
    for chat_state in chat_states:
        if chat_state.get("finished"):
            continue
        assert (
            chat_state["messages"][-1]["role"] == "assistant"
        ), "Expected the last role to be assistant"
        assistant_response = chat_state["messages"][-1]["content"]
        function_calls = extract_json_objects(assistant_response)
        if len(function_calls) == 0:
            chat_state["finished"] = True
    return chat_states


def run_tool_calls(chat_states):
    """
    Execute tool calls found in chat states.
    """
    logger.debug(f"Running tool calls for {len(chat_states)} chat states")
    for chat_state in chat_states:
        if chat_state.get("finished"):
            logger.debug("Chat state already finished, skipping tool calls")
            continue
        assert (
            chat_state["messages"][-1]["role"] == "assistant"
        ), "Expected the last role to be assistant to run tool calls"
        try:
            assistant_response = chat_state["messages"][-1]["content"]
            function_calls = extract_json_objects(assistant_response)
            if len(function_calls) > 1:
                logger.warning("Multiple function calls found in assistant response")
                raise ValueError(
                    "Expected only one function call in assistant response"
                )
            elif len(function_calls) == 1:
                function_call = function_calls[0]
                query = function_call["function"]["parameters"]["query"]
                logger.info(f"Executing search with query: {query}")
                results = search(query, return_type=str, results=2)
                chat_state["messages"].append({"role": "ipython", "content": results})
                logger.debug("Added search results to chat state")
        except Exception as e:
            logger.error(f"Error during tool call: {str(e)}")
            chat_state["messages"].append(
                {"role": "system", "content": f"Error during post-processing: {str(e)}"}
            )
            chat_state["finished"] = True
    return chat_states


def get_mask(text, tokenizer):
    encoding = tokenizer(text, add_special_tokens=False)
    start_header_id = tokenizer.convert_tokens_to_ids("<|start_header_id|>")
    assistant_token = tokenizer.convert_tokens_to_ids("assistant")
    end_header_id = tokenizer.convert_tokens_to_ids("<|end_header_id|>")
    eot_id = tokenizer.convert_tokens_to_ids("<|eot_id|>")
    assistant_ranges = []
    i = 0
    while i < len(encoding.input_ids) - 1:
        if (
            encoding.input_ids[i] == start_header_id
            and encoding.input_ids[i + 1] == assistant_token
        ):
            i += 2
            while (
                i < len(encoding.input_ids) and encoding.input_ids[i] != end_header_id
            ):
                i += 1
            i += 2
            start_idx = i
            while i < len(encoding.input_ids) and encoding.input_ids[i] != eot_id:
                i += 1
            end_idx = i
            assistant_ranges.append((start_idx, end_idx))
        else:
            i += 1
    mask = [0] * len(encoding.input_ids)
    for start_idx, end_idx in assistant_ranges:
        for idx in range(start_idx, end_idx):
            mask[idx] = 1
    return torch.tensor(mask, dtype=torch.int)


def check_exceeded_max_new_tokens(chat_states, max_new_tokens, tokenizer):
    for chat_state in chat_states:
        if chat_state.get("finished"):
            continue
        initial_length = chat_state["initial_length"]
        new_length = get_chat_num_tokens(chat_state, tokenizer)
        if new_length - initial_length > max_new_tokens:
            chat_state["finished"] = True
    return chat_states


@dataclass
class AgenticOutputs:
    prompt_tokens: list[torch.Tensor]
    response_tokens: list[torch.Tensor]
    response_masks: list[torch.Tensor]
    final_response_str: list[str]
    full_chat_states: list[dict]


def get_chat_num_tokens(chat_state, tokenizer):
    chat_text = apply_chat_template(chat_state, tokenizer=tokenizer)["text"]
    return (
        tokenizer(chat_text, add_special_tokens=False, return_tensors="pt")["input_ids"]
        .squeeze()
        .shape[0]
    )


def run_agent(
    generate_fn, tokenizer, questions, max_generations=5, max_new_tokens=4096
):
    """
    Run the agent to completion for a batch of questions.
    """
    logger.info(f"Starting agent run with {len(questions)} questions")
    logger.debug(
        f"Max generations: {max_generations}, Max new tokens: {max_new_tokens}"
    )

    chat_states = [get_initial_chat(q) for q in questions]
    # set the initial_prompt length
    for i, chat_state in enumerate(chat_states):
        chat_state["initial_length"] = get_chat_num_tokens(chat_state, tokenizer)
        logger.debug(f"Initial length for question {i}: {chat_state['initial_length']}")

    # agent loop
    for i in range(max_generations):
        logger.info(f"Starting generation step {i+1}/{max_generations}")
        chat_states = run_agent_generations(generate_fn, tokenizer, chat_states)
        chat_states = check_finished_chats(chat_states)
        chat_states = run_tool_calls(chat_states)
        chat_states = check_exceeded_max_new_tokens(
            chat_states, max_new_tokens, tokenizer
        )
        finished_count = sum(1 for state in chat_states if state.get("finished"))
        logger.info(
            f"Finished {finished_count}/{len(chat_states)} chat states after step {i+1}"
        )

    logger.info("Agent run completed")

    # Process final outputs
    logger.debug("Processing final outputs")
    answers = []
    for chat in chat_states:
        answers.append(chat["messages"][-1]["content"])
        logger.debug(f"Final answer: {chat['messages'][-1]['content'][:100]}...")

    def split_prompt_assistant(convo_text):
        marker = "<|start_header_id|>assistant<|end_header_id|>"
        idx = convo_text.find(marker)
        if idx == -1:
            logger.error("Could not find assistant marker in conversation text")
            raise ValueError("Could not find assistant marker in conversation text.")
            return convo_text, ""
        prompt = convo_text[: idx + len(marker)]
        assistant_response = convo_text[idx + len(marker) :]
        return prompt, assistant_response

    str_chats = [
        apply_chat_template(chat, tokenizer=tokenizer)["text"] for chat in chat_states
    ]
    prompt_toks, response_toks, response_masks = [], [], []

    logger.debug("Processing tokenization")
    for i, str_chat in enumerate(str_chats):
        prompt, response = split_prompt_assistant(str_chat)
        prompt_toks.append(
            tokenizer(prompt, add_special_tokens=False, return_tensors="pt")[
                "input_ids"
            ].squeeze()
        )
        response_toks.append(
            tokenizer(response, add_special_tokens=False, return_tensors="pt")[
                "input_ids"
            ].squeeze()[:max_new_tokens]
        )
        mask = get_mask(str_chat, tokenizer)[len(prompt_toks[-1]) :][:max_new_tokens]
        response_masks.append(mask)
        logger.debug(f"Processed tokens for chat {i}")

    final_response_str = [chat["messages"][-1]["content"] for chat in chat_states]
    full_chat_states = chat_states

    logger.info("Agent run completed successfully")
    return AgenticOutputs(
        prompt_tokens=prompt_toks,
        response_tokens=response_toks,
        response_masks=response_masks,
        final_response_str=final_response_str,
        full_chat_states=full_chat_states,
    )


# Verification
async def verify(student_answer: str, question: str, answer: str) -> bool:
    """
    Verify if student's answer matches the correct answer.

    Args:
        student_answer: The model's answer
        question: The original question
        answer: The ground truth answer

    Returns:
        bool: True if answer is correct, False otherwise
    """
    logger.debug(f"Verifying answer for question: {question}")
    logger.debug(f"Student answer: {student_answer}")
    logger.debug(f"Correct answer: {answer}")

    # Simple string matching for now
    # TODO: Implement more sophisticated matching
    return student_answer.strip().lower() == answer.strip().lower()


def check_student_answers(
    questions: List[str],
    answers: List[str],
    student_answers: List[str],
    vllm_generate_func: Callable[[List[str]], List[str]],
    tokenizer,
    log_file: str = "qa_log.txt",
) -> List[bool]:
    """
    Evaluates a list of student answers against the true answers using a vLLM generate function.
    """
    logger.info(f"Checking {len(questions)} student answers")

    if not (len(questions) == len(answers) == len(student_answers)):
        logger.error(
            "Mismatched lengths between questions, answers, and student answers"
        )
        raise ValueError(
            "The number of questions, answers, and student answers must be equal."
        )

    prompts = []
    for question, answer, student_ans in zip(questions, answers, student_answers):
        prompt_text = (
            "You are grading a student's answer. For the following question, "
            "compare the student's answer to the correct answer. Reply with 'Yes' if the student's answer is correct, or 'No' if it is completely incorrect.\n\n"
            f"Question: {question}\n"
            f"Correct Answer: {answer}\n"
            f"Student Answer: {student_ans}\n"
        )
        formatted_prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt_text}],
            tokenize=False,
            add_generation_prompt=True,
        )
        prompts.append(formatted_prompt)
        logger.debug(f"Created verification prompt for question: {question[:50]}...")

    logger.info("Generating verification responses")
    responses = vllm_generate_func(prompts)
    responses_text = []
    for response in responses:
        if hasattr(response, "outputs"):
            responses_text.append(response.outputs[0].text)
        else:
            responses_text.append(response)
    logger.debug(f"Got {len(responses_text)} verification responses")

    results = []
    for response in responses_text:
        results.append("yes" in response.lower())
        logger.debug(f"Verification result: {'yes' in response.lower()}")

    logger.info(f"Verification complete. {sum(results)}/{len(results)} answers correct")

    # Append the QA details and verifier's response to the specified log file
    with open(log_file, "a") as file:
        for question, answer, student_ans, verifier_response in zip(
            questions, answers, student_answers, responses_text
        ):
            file.write("Question: " + question + "\n")
            file.write("Correct Answer: " + answer + "\n")
            file.write("Student Answer: " + student_ans + "\n")
            file.write("Verifier said: " + verifier_response + "\n")
            file.write("-" * 40 + "\n")

    return results


# Reward Functions
def build_reward_correctness_fn(generate_fn, tokenizer):
    def reward_correctness(prompts, completions, **reward_kwargs):
        teacher_answers = reward_kwargs["answer"]
        student_answers = [
            completion["messages"][-1]["content"] for completion in completions
        ]

        correct = check_student_answers(
            prompts,
            teacher_answers,
            student_answers,
            vllm_generate_func=generate_fn,
            tokenizer=tokenizer,
        )
        return correct

    return reward_correctness


def reward_formatting(prompts, completions, **reward_kwargs):
    # make sure full chats doesn't have any error function calls
    has_error = [False] * len(completions)
    for i, chat in enumerate(completions):
        for message in chat["messages"]:
            if "Error during" in message["content"]:
                has_error[i] = True
                break
    return [0.7 if not e else 0 for e in has_error]


def reward_retry_behavior(completions: list[dict], **reward_kwargs) -> list[float]:
    """
    Reward function that encourages optimal retry behavior by counting total function calls
    across all assistant messages in the conversation.
    """
    rewards: list[float] = []

    for completion in completions:
        # Get ALL assistant messages
        assistant_msgs: list[str] = [
            msg["content"]
            for msg in completion["messages"]
            if msg["role"] == "assistant" and msg["content"] is not None
        ]

        if not assistant_msgs:
            rewards.append(0.0)
            continue

        # Count total function calls across all messages
        total_retries: int = 0
        for msg in assistant_msgs:
            total_retries += len(extract_json_objects(msg))

        # Calculate reward using modified sigmoid function
        x: float = float(total_retries - 4)  # Center peak at 4 retries
        base_reward: float = 1.0 / (1.0 + np.exp(-x + abs(x) / 2))

        # Additional penalty for excessive retries
        if total_retries > 6:
            penalty: float = 0.2 * (total_retries - 6)
            base_reward = max(0.1, base_reward - penalty)

        rewards.append(base_reward)

    return rewards


def reward_exact_match_chunk_query(prompts, completions, **reward_kwargs):
    """
    Reward function that checks if the model's search queries hit the correct chunk content.
    """
    logger.debug(f"Calculating rewards for {len(prompts)} prompts")

    # Get correct chunk contents from reward kwargs
    correct_contents = reward_kwargs.get("chunk_content", [])
    if not correct_contents:
        logger.error("No chunk_content provided in reward_kwargs")
        raise ValueError("chunk_content must be provided in reward_kwargs")

    rewards = []
    for i, (chat_state, correct_content) in enumerate(
        zip(completions, correct_contents)
    ):
        # Get all ipython messages (search results) from the chat
        search_results = [
            msg["content"] for msg in chat_state["messages"] if msg["role"] == "ipython"
        ]
        logger.debug(f"Found {len(search_results)} search results for prompt {i}")

        # Check if any search hit the correct chunk content
        found_correct_chunk = False
        for result in search_results:
            if correct_content in result:
                found_correct_chunk = True
                logger.debug(
                    f"Found correct chunk content in search results for prompt {i}"
                )
                break

        reward = 1.0 if found_correct_chunk else 0.0
        rewards.append(reward)
        logger.debug(f"Reward for prompt {i}: {reward}")

    logger.info(f"Average reward: {sum(rewards)/len(rewards):.3f}")
    return rewards


def run_eval(generate_fn, verify_fn, tokenizer):
    logger.info("Starting evaluation")
    train_dataset, test_dataset = get_qa_dataset()
    questions = test_dataset["prompt"]
    logger.info(f"Loaded {len(questions)} test questions")

    agentic_outputs = run_agent(generate_fn, tokenizer, questions)
    full_chat_states = agentic_outputs.full_chat_states
    final_responses = agentic_outputs.final_response_str

    logger.info("Calculating rewards")
    rewards = verify_fn(questions, full_chat_states, answer=test_dataset["answer"])
    avg_reward = sum(rewards) / len(rewards)

    logger.info("EVALUATION RESULTS:")
    logger.info(f"Percentage of correct answers: {avg_reward:.3f}")
    logger.info("=" * 30)

    return full_chat_states
