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

import nest_asyncio
import torch

from search_module import get_qa_dataset, search

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

You may also reason in any message, thinking step by step about how to answer the question. Wrap your reasoning in <reasoning> and </reasoning> tags.

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
    Removes all content between <reasoning> and </reasoning> tags,
    including the tags themselves.

    Parameters:
        text (str): The input text that may contain <reasoning>...</reasoning> tags.

    Returns:
        str: The text with the tags and their content removed.
    """
    # The regex pattern matches from <reasoning> to </reasoning> non-greedily.
    pattern = r"<reasoning>.*?</reasoning>"
    cleaned_text = re.sub(pattern, "", text, flags=re.DOTALL)
    return cleaned_text


def run_agent_generations(generate_fn, tokenizer, chat_states):
    """
    Run generation for chat states requiring assistant responses.

    Args:
        generate_fn: Function to generate responses
        tokenizer: Tokenizer for processing text
        chat_states: List of chat states

    Returns:
        list: Updated chat states
    """
    prompts = []
    batch_indices = []
    # Prepare prompts for chat states needing an assistant response.
    for idx, chat_state in enumerate(chat_states):
        if chat_state.get("finished"):
            continue

        if chat_state["messages"][-1]["role"] in ["ipython", "user"]:
            prompt = apply_chat_template(chat_state, tokenizer=tokenizer)["text"]
            prompts.append(prompt)
            batch_indices.append(idx)

    if prompts:
        responses = generate_fn(prompts)
        for i, idx in enumerate(batch_indices):
            chat_state = chat_states[idx]
            full_response = responses[i].outputs[0].text
            assistant_response = full_response.split(
                "<|start_header_id|>assistant<|end_header_id|>"
            )[-1]
            chat_state["messages"].append(
                {"role": "assistant", "content": assistant_response}
            )
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

    Args:
        chat_states: List of chat states

    Returns:
        list: Updated chat states with tool call results
    """
    for chat_state in chat_states:
        if chat_state.get("finished"):
            continue
        assert (
            chat_state["messages"][-1]["role"] == "assistant"
        ), "Expected the last role to be assistant to run tool calls"
        try:
            assistant_response = chat_state["messages"][-1]["content"]
            function_calls = extract_json_objects(assistant_response)
            if len(function_calls) > 1:
                raise ValueError(
                    "Expected only one function call in assistant response"
                )
            elif len(function_calls) == 1:
                function_call = function_calls[0]
                query = function_call["function"]["parameters"]["query"]
                results = search(query, return_type=str, results=2)
                chat_state["messages"].append({"role": "ipython", "content": results})
        except Exception as e:
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

    Args:
        generate_fn: Function to generate model responses
        tokenizer: Tokenizer for processing text
        batch: Batch of data containing questions
        max_generations: Maximum number of generation steps

    Returns:
        list: Final answers for each question
    """
    chat_states = [get_initial_chat(q) for q in questions]
    # set the initial_prompt length
    for chat_state in chat_states:
        chat_state["initial_length"] = get_chat_num_tokens(chat_state, tokenizer)

    # agent loop
    for i in range(max_generations):
        chat_states = run_agent_generations(generate_fn, tokenizer, chat_states)
        chat_states = check_finished_chats(chat_states)
        chat_states = run_tool_calls(chat_states)
        chat_states = check_exceeded_max_new_tokens(
            chat_states, max_new_tokens, tokenizer
        )

    answers = []
    for chat in chat_states:
        answers.append(chat["messages"][-1]["content"])

    def split_prompt_assistant(convo_text):
        marker = "<|start_header_id|>assistant<|end_header_id|>"
        idx = convo_text.find(marker)
        if idx == -1:
            raise ValueError("Could not find assistant marker in conversation text.")
            return convo_text, ""
        # Include the marker in the prompt by slicing up to the end of the marker.
        prompt = convo_text[: idx + len(marker)]
        # The assistant response is everything after the marker.
        assistant_response = convo_text[idx + len(marker) :]
        return prompt, assistant_response

    str_chats = [
        apply_chat_template(chat, tokenizer=tokenizer)["text"] for chat in chat_states
    ]
    prompt_toks, response_toks, response_masks = [], [], []
    for str_chat in str_chats:
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

    final_response_str = [chat["messages"][-1]["content"] for chat in chat_states]
    full_chat_states = chat_states
    agentic_outputs = AgenticOutputs(
        prompt_tokens=prompt_toks,
        response_tokens=response_toks,
        response_masks=response_masks,
        final_response_str=final_response_str,
        full_chat_states=full_chat_states,
    )

    return agentic_outputs


# Verification
async def check_correctness(question, student_answer, answer):
    """
    Calculate reward for a given student answer.

    Args:
        question (str): The original question
        student_answer (str): The model's answer
        answer (str): The ground truth answer

    Returns:
        float: Reward value (1 for correct, 0 for incorrect)
    """
    # log to "./reward_func.log"
    with open("reward_func.log", "a") as f:
        f.write("\n" + "==" * 40 + "\n\n")
        f.write(f"Question: {question}\n")
        f.write(f"Student Answer: {student_answer}\n")
        f.write(f"Answer: {answer}\n")
        if student_answer.startswith("Error during"):
            f.write(f"failed function call")
            return 0
        if len(student_answer) < 5:
            f.write(f"failed Too short answer\n")
            return 0
        else:
            f.write(f"last message didn't fail\n")
            student_answer_clean = remove_reasoning(student_answer)
            is_correct = await verify(student_answer_clean, question, answer)
            f.write(f"Is Correct: {is_correct}, so reward is {int(is_correct)}\n")
            return 1 if is_correct else 0


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
    The function applies the chat template to each prompt before passing it to the generate function.
    It also appends the details of each QA pair and the verifier's response to a log file.

    Args:
        questions: A list of strings representing the questions.
        answers: A list of strings representing the correct answers.
        student_answers: A list of strings containing the student's answers.
        vllm_generate_func: A function that takes a list of chat-formatted prompt strings and returns a list of generated outputs.
        tokenizer: The tokenizer used to apply the chat template.
        log_file: Optional; path to the file where the QA pairs and verification responses will be appended.

    Returns:
        A list of booleans indicating whether each student's answer is correct.
    """
    if not (len(questions) == len(answers) == len(student_answers)):
        raise ValueError(
            "The number of questions, answers, and student answers must be equal."
        )

    prompts = []
    for question, answer, student_ans in zip(questions, answers, student_answers):
        # Construct the plain text prompt for each QA pair.
        prompt_text = (
            "You are grading a student's answer. For the following question, "
            "compare the student's answer to the correct answer. Reply with 'Yes' if the student's answer is correct, or 'No' if it is completely incorrect.\n\n"
            f"Question: {question}\n"
            f"Correct Answer: {answer}\n"
            f"Student Answer: {student_ans}\n"
        )
        # Apply the chat template to the prompt.
        formatted_prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt_text}],
            tokenize=False,
            add_generation_prompt=True,
        )
        prompts.append(formatted_prompt)

    # Get the model responses in batch (each response should ideally be "Yes" or "No")
    responses = vllm_generate_func(prompts)
    responses_text = [response.outputs[0].text for response in responses]

    # Evaluate each response and mark as correct if "yes" appears in the answer (case-insensitive)
    results = []
    for response in responses_text:
        results.append("yes" in response.lower())

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


def run_eval(generate_fn, verify_fn, tokenizer):
    train_dataset, test_dataset = get_qa_dataset()
    questions = test_dataset["prompt"]
    agentic_outputs = run_agent(generate_fn, tokenizer, questions)
    full_chat_states = agentic_outputs.full_chat_states
    final_responses = agentic_outputs.final_response_str
    rewards = verify_fn(questions, full_chat_states, answer=test_dataset["answer"])

    print("RESULTS:")
    print("percentage of correct answers:", sum(rewards) / len(rewards))
    print("=" * 30)

    return full_chat_states
