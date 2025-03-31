"""
RL helpers module for handling tool-based conversations.
This module provides utility functions for handling chat-based tool interactions
and calculating rewards based on the quality of responses.
"""

import inspect
import json
import re
from dataclasses import dataclass
from datetime import datetime

import nest_asyncio
import numpy as np
import torch

from src.config import log_metric, logger
from src.search_module import get_qa_dataset, search

# Apply nest_asyncio for supporting async operations in notebooks
nest_asyncio.apply()

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
            assistant_response = full_response.split("<|start_header_id|>assistant<|end_header_id|>")[-1]
            chat_state["messages"].append({"role": "assistant", "content": assistant_response})
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
        assert chat_state["messages"][-1]["role"] == "assistant", "Expected the last role to be assistant"
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
    total_retries = 0

    for chat_state in chat_states:
        if chat_state.get("finished"):
            logger.debug("Chat state already finished, skipping tool calls")
            continue
        assert chat_state["messages"][-1]["role"] == "assistant", (
            "Expected the last role to be assistant to run tool calls"
        )
        try:
            assistant_response = chat_state["messages"][-1]["content"]
            function_calls = extract_json_objects(assistant_response)
            if len(function_calls) > 1:
                logger.warning("Multiple function calls found in assistant response")
                raise ValueError("Expected only one function call in assistant response")
            elif len(function_calls) == 1:
                function_call = function_calls[0]
                query = function_call["function"]["parameters"]["query"]
                logger.info(f"🔍 Search Query: {query}")
                results = search(query, return_type=str, results=2)
                chat_state["messages"].append({"role": "ipython", "content": results})

                # Count retries
                retries = len(extract_json_objects(assistant_response))
                total_retries += retries

                logger.debug("Added search results to chat state")
        except Exception as e:
            logger.error(f"Error during tool call: {str(e)}")
            chat_state["messages"].append({"role": "system", "content": f"Error during post-processing: {str(e)}"})
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
        if encoding.input_ids[i] == start_header_id and encoding.input_ids[i + 1] == assistant_token:
            i += 2
            while i < len(encoding.input_ids) and encoding.input_ids[i] != end_header_id:
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
    return tokenizer(chat_text, add_special_tokens=False, return_tensors="pt")["input_ids"].squeeze().shape[0]


def run_agent(
    generate_fn,
    tokenizer,
    questions,
    max_generations=5,
    max_new_tokens=4096,
    correct_contents=None,
):
    """
    Run the agent to completion for a batch of questions.
    """
    logger.info(f"Starting agent run with {len(questions)} questions")
    logger.debug(f"Max generations: {max_generations}, Max new tokens: {max_new_tokens}")

    chat_states = [get_initial_chat(q) for q in questions]
    # Add correct content to chat states if provided
    if correct_contents:
        for chat_state, correct_content in zip(chat_states, correct_contents):
            chat_state["correct_content"] = correct_content

    # set the initial_prompt length
    for i, chat_state in enumerate(chat_states):
        chat_state["initial_length"] = get_chat_num_tokens(chat_state, tokenizer)
        logger.debug(f"Initial length for question {i}: {chat_state['initial_length']}")

    # agent loop
    for i in range(max_generations):
        logger.info(f"Starting generation step {i + 1}/{max_generations}")
        chat_states = run_agent_generations(generate_fn, tokenizer, chat_states)
        chat_states = check_finished_chats(chat_states)
        chat_states = run_tool_calls(chat_states)
        chat_states = check_exceeded_max_new_tokens(chat_states, max_new_tokens, tokenizer)
        finished_count = sum(1 for state in chat_states if state.get("finished"))
        logger.info(f"Finished {finished_count}/{len(chat_states)} chat states after step {i + 1}")

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

    str_chats = [apply_chat_template(chat, tokenizer=tokenizer)["text"] for chat in chat_states]
    prompt_toks, response_toks, response_masks = [], [], []

    logger.debug("Processing tokenization")
    for i, str_chat in enumerate(str_chats):
        prompt, response = split_prompt_assistant(str_chat)
        prompt_toks.append(tokenizer(prompt, add_special_tokens=False, return_tensors="pt")["input_ids"].squeeze())
        response_toks.append(
            tokenizer(response, add_special_tokens=False, return_tensors="pt")["input_ids"].squeeze()[:max_new_tokens]
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
    questions: list[str],
    answers: list[str],
    student_answers: list,  # Can be strings or dicts
    vllm_generate_func,
    tokenizer,
    log_file=None,
) -> list[bool]:
    """
    Evaluates a list of student answers against the true answers using a vLLM generate function.

    Args:
        questions: List of questions
        answers: List of correct answers
        student_answers: List of student answers to evaluate
        vllm_generate_func: Function to generate verification responses
        tokenizer: Tokenizer for formatting prompts
        log_file: Optional path to write detailed results

    Returns:
        List of boolean results (True for correct answers)
    """
    logger.info(f"Checking {len(questions)} student answers")

    if not (len(questions) == len(answers) == len(student_answers)):
        logger.error("Mismatched lengths between questions, answers, and student answers")
        raise ValueError("The number of questions, answers, and student answers must be equal.")

    prompts = []
    for question, answer, student_ans in zip(questions, answers, student_answers):
        prompt_text = (
            "You are grading a student's answer to a question. For the following question, "
            "compare the student's answer to the correct answer. Reply with 'Yes' if the student's answer contains the correct information, "
            "even if it's not an exact match. If the student's answer doesn't contain the right information or is completely incorrect, reply with 'No'.\n\n"
            f"Question: {question}\n"
            f"Correct Answer: {answer}\n"
            f"Student Answer: {student_ans}\n\n"
            "Your response should be just 'Yes' or 'No'."
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
        # Handle different response formats
        if hasattr(response, "outputs"):
            try:
                responses_text.append(response.outputs[0].text)
            except (AttributeError, IndexError):
                # Fallback for simple string responses
                responses_text.append(str(response))
        else:
            responses_text.append(str(response))
    logger.debug(f"Got {len(responses_text)} verification responses")

    results = []
    for response in responses_text:
        results.append("yes" in response.lower())
        logger.debug(f"Verification result: {'yes' in response.lower()}")

    logger.info(f"Verification complete. {sum(results)}/{len(results)} answers correct")

    # Append the QA details and verifier's response to the specified log file
    if log_file:
        with open(log_file, "a") as file:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            file.write(f"\n📝 === QA Evaluation at {timestamp} ===\n")
            file.write(f"📂 File: {__file__}\n")

            # Get current frame info safely
            frame = inspect.currentframe()
            if frame:
                file.write(f"📍 Line: {frame.f_lineno}\n")
                # Don't forget to delete the frame to avoid reference cycles
                del frame

            file.write("=" * 80 + "\n")

            for i, (question, answer, student_ans, verifier_response) in enumerate(
                zip(questions, answers, student_answers, responses_text)
            ):
                file.write(f"\n❓ Question {i + 1}:\n")
                file.write("-" * 40 + "\n")
                file.write(f"📋 Question: {question}\n")
                file.write(f"✅ Correct Answer: {answer}\n")
                file.write(f"👨‍🎓 Student Answer: {student_ans}\n")
                file.write(f"🔍 Verifier said: {verifier_response}\n")

                # Add search results if available in the chat state
                if isinstance(student_ans, dict) and "messages" in student_ans:
                    # Get messages from dict
                    messages = student_ans.get("messages", [])
                    search_results = [msg.get("content", "") for msg in messages if msg.get("role") == "ipython"]
                    if search_results:
                        file.write("\n🔎 Search Results:\n")
                        for j, result in enumerate(search_results, 1):
                            file.write(f"\nSearch {j}:\n{result}\n")

                file.write("-" * 40 + "\n")

            file.write(
                f"\n📊 Summary: {sum(results)}/{len(results)} answers correct ({sum(results) / len(results) * 100:.2f}%)\n"
            )
            file.write("=" * 80 + "\n\n")

    return results


# Reward Functions
def build_reward_correctness_fn(generate_fn, tokenizer, log_file=None):
    def reward_correctness(prompts, completions, **reward_kwargs):
        teacher_answers = reward_kwargs["answer"]
        student_answers = [completion["messages"][-1]["content"] for completion in completions]

        # Log non-exact matches
        for i, (student, teacher) in enumerate(zip(student_answers, teacher_answers)):
            if student.strip().lower() != teacher.strip().lower():
                logger.warning(f"Non-exact match at index {i}:\nStudent: {student}\nTeacher: {teacher}")

        correct = check_student_answers(
            prompts,
            teacher_answers,
            student_answers,
            vllm_generate_func=generate_fn,
            tokenizer=tokenizer,
            log_file=log_file,
        )

        # Log correctness metrics with length info
        log_metric("rewards/correctness", np.mean(correct), reward_kwargs.get("step", 0))
        log_metric("rewards/correctness_std", np.std(correct), reward_kwargs.get("step", 0))

        # Log length metrics
        student_lengths = [len(ans.strip()) for ans in student_answers]
        teacher_lengths = [len(ans.strip()) for ans in teacher_answers]
        log_metric(
            "metrics/avg_student_length",
            np.mean(student_lengths),
            reward_kwargs.get("step", 0),
        )
        log_metric(
            "metrics/avg_teacher_length",
            np.mean(teacher_lengths),
            reward_kwargs.get("step", 0),
        )
        log_metric(
            "metrics/length_ratio",
            np.mean(student_lengths) / np.mean(teacher_lengths),
            reward_kwargs.get("step", 0),
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
                logger.warning(f"Error in chat {i}: {message['content']}")
                break

    rewards = [0.7 if not e else 0 for e in has_error]

    # Log formatting metrics
    log_metric("rewards/formatting", np.mean(rewards), reward_kwargs.get("step", 0))
    log_metric("rewards/formatting_std", np.std(rewards), reward_kwargs.get("step", 0))
    log_metric("metrics/error_rate", np.mean(has_error), reward_kwargs.get("step", 0))

    return rewards


def reward_retry_behavior(completions: list[dict], **reward_kwargs) -> list[float]:
    """
    Reward function that encourages optimal retry behavior by only rewarding completions
    where every assistant message contains at most 1 JSON object.
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

        # Check if every message has at most 1 JSON object
        has_multiple_json = False
        total_json_objects = 0

        for msg in assistant_msgs:
            json_objects = extract_json_objects(msg)
            json_count = len(json_objects)
            total_json_objects += json_count

            if json_count > 1:
                has_multiple_json = True
                logger.warning(f"Message contains {json_count} JSON objects, which exceeds the limit of 1")
                break

        # Only reward if no message has multiple JSON objects
        if has_multiple_json:
            rewards.append(0.0)
        else:
            # Base reward is 1.0 if constraint is met
            base_reward = 1.0

            # Slight penalty for having too many total JSON objects across all messages
            if total_json_objects > 4:
                penalty = 0.1 * (total_json_objects - 4)
                base_reward = max(0.2, base_reward - penalty)
                logger.debug(f"Applied penalty for {total_json_objects} total JSON objects: {penalty}")

            rewards.append(base_reward)

    # Log retry behavior metrics
    log_metric("rewards/retry_behavior", np.mean(rewards), reward_kwargs.get("step", 0))
    log_metric("rewards/retry_behavior_std", np.std(rewards), reward_kwargs.get("step", 0))
    log_metric(
        "metrics/avg_json_per_msg",
        np.mean(
            [
                len(extract_json_objects(msg["content"]))
                for completion in completions
                for msg in completion["messages"]
                if msg["role"] == "assistant"
            ]
        ),
        reward_kwargs.get("step", 0),
    )
    log_metric(
        "metrics/multiple_json_violation_rate",
        np.mean([0.0 if rewards[i] > 0.0 else 1.0 for i in range(len(rewards))]),
        reward_kwargs.get("step", 0),
    )

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
    for i, (chat_state, correct_content) in enumerate(zip(completions, correct_contents)):
        # Get all ipython messages (search results) from the chat
        search_results = [msg["content"] for msg in chat_state["messages"] if msg["role"] == "ipython"]
        logger.debug(f"Found {len(search_results)} search results for prompt {i}")

        # Log ground truth chunk and searched chunks
        logger.info(f"📝 Ground Truth Chunk: {correct_content}")
        for j, result in enumerate(search_results):
            logger.info(f"🔍 Searched Chunk {j + 1}: {result}")

        # Check if any search hit the correct chunk content
        found_correct_chunk = False
        for result in search_results:
            if correct_content in result:
                found_correct_chunk = True
                logger.debug(f"Found correct chunk content in search results for prompt {i}")
                break

        if not found_correct_chunk:
            logger.warning(
                f"Failed to find correct chunk for prompt {i}:\n"
                f"Search results: {[r[:100] + '...' for r in search_results]}"
            )

        reward = 1.0 if found_correct_chunk else 0.0
        rewards.append(reward)
        logger.debug(f"Reward for prompt {i}: {reward}")

        # Log detailed metrics for debugging
        log_metric(
            f"debug/chunk_match_{i}",
            1 if found_correct_chunk else 0,
            reward_kwargs.get("step", 0),
        )
        log_metric(
            f"debug/search_results_count_{i}",
            len(search_results),
            reward_kwargs.get("step", 0),
        )
        if search_results:
            log_metric(
                f"debug/result_length_{i}",
                np.mean([len(r.split()) for r in search_results]),
                reward_kwargs.get("step", 0),
            )

    # Log chunk query metrics
    log_metric("rewards/chunk_query", np.mean(rewards), reward_kwargs.get("step", 0))
    log_metric("rewards/chunk_query_std", np.std(rewards), reward_kwargs.get("step", 0))
    log_metric(
        "metrics/avg_search_results",
        np.mean(
            [
                len([msg["content"] for msg in chat_state["messages"] if msg["role"] == "ipython"])
                for chat_state in completions
            ]
        ),
        reward_kwargs.get("step", 0),
    )
    log_metric("metrics/chunk_match_rate", np.mean(rewards), reward_kwargs.get("step", 0))

    # Log detailed debugging info
    logger.info("Chunk Query Rewards Summary:")
    logger.info(f"Total prompts: {len(prompts)}")
    logger.info(f"Correct matches: {sum(rewards)}")
    logger.info(f"Average reward: {np.mean(rewards):.3f}")
    logger.info(f"Reward std: {np.std(rewards):.3f}")

    return rewards


def run_eval(generate_fn, verify_fn, tokenizer, output_file=None, debug_file=None):
    """
    Run evaluation on the test dataset and return results.

    Args:
        generate_fn: Function to generate completions
        verify_fn: Function to verify results
        tokenizer: Tokenizer for processing text
        output_file: Path to save evaluation results summary
        debug_file: Path to save detailed debug information

    Returns:
        full_chat_states: The chat states from evaluation
    """
    train_dataset, test_dataset = get_qa_dataset()
    questions = test_dataset["prompt"]
    agentic_outputs = run_agent(generate_fn, tokenizer, questions)
    full_chat_states = agentic_outputs.full_chat_states
    final_responses = agentic_outputs.final_response_str
    rewards = verify_fn(questions, full_chat_states, answer=test_dataset["answer"])

    # Calculate results
    percent_correct = sum(rewards) / len(rewards) * 100

    # Log results to console
    logger.info("RESULTS:")
    logger.info(f"percentage of correct answers: {percent_correct:.2f}%")
    logger.info("=" * 30)

    # Save results to file if specified
    if output_file:
        try:
            with open(output_file, "w") as f:
                f.write("EVALUATION RESULTS\n")
                f.write("=================\n\n")
                f.write(f"Total questions: {len(questions)}\n")
                f.write(f"Correct answers: {sum(rewards)}\n")
                f.write(f"Percentage correct: {percent_correct:.2f}%\n\n")

                f.write("Individual results:\n")
                for i, (q, r, resp) in enumerate(zip(questions, rewards, final_responses)):
                    f.write(f"\nQ{i + 1}: {q[:100]}...\n")
                    f.write(f"Correct: {'✓' if r else '✗'}\n")
                    f.write(f"Response: {resp[:150]}...\n")
                    f.write("-" * 40 + "\n")
            logger.info(f"Saved evaluation results to {output_file}")
        except Exception as e:
            logger.error(f"Error saving results file: {e}")

    # Save debug information if specified
    if debug_file:
        try:
            import json

            debug_data = []
            for i, (q, r, resp, chat) in enumerate(zip(questions, rewards, final_responses, full_chat_states)):
                debug_data.append(
                    {
                        "question_id": i,
                        "question": q,
                        "is_correct": bool(r),
                        "final_response": resp,
                        "chat_state": {
                            k: str(v) if isinstance(v, (list, dict)) else v
                            for k, v in chat.items()
                            if k != "tokenizer"
                        },
                    }
                )

            with open(debug_file, "w") as f:
                json.dump(debug_data, f, indent=2)
            logger.info(f"Saved debug information to {debug_file}")
        except Exception as e:
            logger.error(f"Error saving debug file: {e}")

    return full_chat_states
