"""
Core agent functionality for handling tool-based conversations.
This module provides a base agent class for handling tool-based conversations.
"""

import re
from dataclasses import dataclass

import torch
from trl.trainer.grpo_trainer import apply_chat_template

from src.config import logger
from src.prompts import build_user_prompt, get_system_prompt
from src.search_module import search
from src.tokenizer_adapter import TokenizerAdapter


@dataclass
class AgenticOutputs:
    """Outputs from running the agent on a batch of questions."""

    prompt_tokens: list[torch.Tensor]
    response_tokens: list[torch.Tensor]
    response_masks: list[torch.Tensor]
    final_response_str: list[str]
    full_chat_states: list[dict]


class Agent:
    """Base agent class for handling tool-based conversations."""

    def __init__(self, tokenizer_adapter: TokenizerAdapter):
        """Initialize the agent with a tokenizer adapter."""
        self.tokenizer_adapter = tokenizer_adapter

    def get_initial_chat(self, question: str) -> dict:
        """Initialize a chat state with the question."""
        return {
            "messages": [
                {"role": "system", "content": get_system_prompt()},
                {"role": "user", "content": build_user_prompt(question)},
            ]
        }

    def extract_search_query(self, text: str) -> str | None:
        """Extract search query from text between <search> tags."""
        pattern = re.compile(r"<search>(.*?)</search>", re.DOTALL)
        matches = pattern.findall(text)
        return matches[-1] if matches else None

    def run_agent_generations(self, generate_fn, tokenizer, chat_states: list[dict]) -> list[dict]:
        """Run generation for chat states requiring assistant responses."""
        logger.debug(f"Starting generation for {len(chat_states)} chat states")
        prompts = []
        batch_indices = []

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

                assistant_response = full_response.split(self.tokenizer_adapter.get_assistant_marker())[-1]
                chat_state["messages"].append({"role": "assistant", "content": assistant_response})
                logger.debug(f"Added assistant response to chat state {idx}")
        else:
            logger.debug("No prompts to generate responses for")

        return chat_states

    def check_finished_chats(self, chat_states: list[dict]) -> list[dict]:
        """Check which chat states are finished (no more search queries)."""
        for chat_state in chat_states:
            if chat_state.get("finished"):
                continue
            assert chat_state["messages"][-1]["role"] == "assistant", "Expected the last role to be assistant"
            assistant_response = chat_state["messages"][-1]["content"]
            if not re.search(r"<search>.*?</search>", assistant_response, re.DOTALL):
                chat_state["finished"] = True
        return chat_states

    def run_tool_calls(self, chat_states: list[dict]) -> list[dict]:
        """Execute tool calls found in chat states."""
        logger.debug(f"Running tool calls for {len(chat_states)} chat states")

        for chat_state in chat_states:
            if chat_state.get("finished"):
                logger.debug("Chat state already finished, skipping tool calls")
                continue

            assert chat_state["messages"][-1]["role"] == "assistant", (
                "Expected the last role to be assistant to run tool calls"
            )
            try:
                assistant_response = chat_state["messages"][-1]["content"]
                search_query = self.extract_search_query(assistant_response)
                if search_query:
                    logger.info(f"🔍 Search Query: {search_query}")
                    results = search(search_query, return_type=str, results=2)
                    formatted_results = f"<information>{results}</information>"
                    logger.info(f"ℹ️ Information: {formatted_results}")

                    # chat_state["messages"].append({"role": "ipython", "content": formatted_results})
                    chat_state["messages"].append({"role": "user", "content": formatted_results})
                    logger.debug("Added search results to chat state")
            except Exception as e:
                logger.error(f"Error during tool call: {str(e)}")
                chat_state["messages"].append({"role": "system", "content": f"Error during post-processing: {str(e)}"})
                chat_state["finished"] = True

        return chat_states

    def get_chat_num_tokens(self, chat_state: dict, tokenizer) -> int:
        """Get number of tokens in chat state."""
        chat_text = apply_chat_template(chat_state, tokenizer=tokenizer)["text"]
        return tokenizer(chat_text, add_special_tokens=False, return_tensors="pt")["input_ids"].squeeze().shape[0]

    def check_exceeded_max_new_tokens(self, chat_states: list[dict], max_new_tokens: int, tokenizer) -> list[dict]:
        """Check if any chat state has exceeded max new tokens."""
        for chat_state in chat_states:
            if chat_state.get("finished"):
                continue
            initial_length = chat_state["initial_length"]
            new_length = self.get_chat_num_tokens(chat_state, tokenizer)
            if new_length - initial_length > max_new_tokens:
                chat_state["finished"] = True
        return chat_states

    def run_agent(
        self,
        generate_fn,
        tokenizer,
        questions: list[str],
        max_generations: int = 5,
        max_new_tokens: int = 4096,
        correct_contents: list[str] | None = None,
    ) -> AgenticOutputs:
        """Run the agent to completion for a batch of questions.

        This method follows the same flow as rl_helpers.py:
        1. Initialize chat states with questions
        2. Run agent loop (generations, check finished, tool calls, check tokens)
        3. Process final outputs (split prompt/response, get masks)

        The key difference from our previous implementation is in how we handle
        the final tokenization and masking, which now matches rl_helpers.py exactly.
        """
        # Step 1: Initialize chat states with questions
        chat_states = [self.get_initial_chat(q) for q in questions]
        if correct_contents:
            for chat_state, correct_content in zip(chat_states, correct_contents):
                chat_state["correct_content"] = correct_content

        # Set initial token lengths for each chat state
        for chat_state in chat_states:
            chat_state["initial_length"] = self.get_chat_num_tokens(chat_state, tokenizer)

        # Step 2: Run agent loop
        for i in range(max_generations):
            chat_states = self.run_agent_generations(generate_fn, tokenizer, chat_states)
            chat_states = self.check_finished_chats(chat_states)
            chat_states = self.run_tool_calls(chat_states)
            chat_states = self.check_exceeded_max_new_tokens(chat_states, max_new_tokens, tokenizer)

        # Step 3: Process final outputs
        # Get the final answers from each chat state
        answers = [chat["messages"][-1]["content"] for chat in chat_states]

        # Convert chat states to text format for tokenization
        str_chats = [apply_chat_template(chat, tokenizer=tokenizer)["text"] for chat in chat_states]
        prompt_toks, response_toks, response_masks = [], [], []

        # Process each chat state to get tokens and masks
        for str_chat in str_chats:
            try:
                # Split into prompt and response parts
                # Note: If assistant marker is missing, split_prompt_assistant will return (full_text, "")
                prompt_text, response_text = self.tokenizer_adapter.split_prompt_assistant(str_chat)

                # Get prompt tokens
                prompt_toks.append(
                    tokenizer(prompt_text, add_special_tokens=False, return_tensors="pt")["input_ids"].squeeze()
                )

                # Get response tokens (truncated to max_new_tokens)
                response_toks.append(
                    tokenizer(response_text, add_special_tokens=False, return_tensors="pt")["input_ids"].squeeze()[
                        :max_new_tokens
                    ]
                )

                # Get full mask and slice it properly
                # This matches rl_helpers.py exactly:
                # 1. Get full mask for entire text
                # 2. Slice from prompt length to end
                # 3. Truncate to max_new_tokens
                full_mask = self.tokenizer_adapter.get_mask(str_chat, tokenizer)
                prompt_len = prompt_toks[-1].shape[0]
                mask = full_mask[prompt_len:][:max_new_tokens]
                response_masks.append(mask)

                # debug if the tokens and masks are of same length by logger info
                logger.debug(f"Prompt tokens length: {len(prompt_toks[-1])}")
                logger.debug(f"Mask length: {len(mask)}")
                logger.debug(f"Response tokens length: {len(response_toks[-1])}")

            except Exception:
                # If anything fails, add empty tensors
                # This matches rl_helpers.py's behavior of not handling errors explicitly
                prompt_toks.append(torch.tensor([], dtype=torch.long))
                response_toks.append(torch.tensor([], dtype=torch.long))
                response_masks.append(torch.tensor([], dtype=torch.long))

        # Return final outputs
        return AgenticOutputs(
            prompt_tokens=prompt_toks,
            response_tokens=response_toks,
            response_masks=response_masks,
            final_response_str=answers,
            full_chat_states=chat_states,
        )
