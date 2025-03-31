"""
Simple CLI inference script with search functionality.

This script allows interaction with a model (with optional LoRA adapter)
and provides search functionality for data retrieval.
"""

import argparse
import json
import os
import re
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

from unsloth import FastLanguageModel
from vllm import SamplingParams

from src.config import MODEL_NAME, OUTPUT_DIR, logger
from src.search_module import load_vectorstore, search


def find_latest_checkpoint(search_dir=None):
    """
    Find the latest checkpoint in the specified directory or OUTPUT_DIR.

    Args:
        search_dir: Directory to search for checkpoints (default: OUTPUT_DIR)

    Returns:
        Path to the latest checkpoint or None if no checkpoints found
    """
    if search_dir is None:
        search_dir = "trainer_output_meta-llama_Llama-3.1-8B-Instruct_gpu1_20250326_134236"
        logger.info(f"No search directory provided, using default: {search_dir}")
    else:
        logger.info(f"Searching for checkpoints in: {search_dir}")

    # Check if the directory exists first
    if not os.path.exists(search_dir):
        logger.warning(f"Search directory {search_dir} does not exist")
        return None

    # First try to find checkpoints in the format checkpoint-{step}
    import glob

    checkpoints = glob.glob(os.path.join(search_dir, "checkpoint-*"))

    if checkpoints:
        # Extract checkpoint numbers and sort
        checkpoint_numbers = []
        for checkpoint in checkpoints:
            match = re.search(r"checkpoint-(\d+)$", checkpoint)
            if match:
                checkpoint_numbers.append((int(match.group(1)), checkpoint))

        if checkpoint_numbers:
            # Sort by checkpoint number (descending)
            checkpoint_numbers.sort(reverse=True)
            latest = checkpoint_numbers[0][1]
            logger.info(f"Found latest checkpoint: {latest}")
            return latest

    # If no checkpoints found, look for saved_adapter_{timestamp}.bin files
    adapter_files = glob.glob(os.path.join(search_dir, "saved_adapter_*.bin"))
    if adapter_files:
        # Sort by modification time (newest first)
        adapter_files.sort(key=os.path.getmtime, reverse=True)
        latest = adapter_files[0]
        logger.info(f"Found latest adapter file: {latest}")
        return latest

    # If all else fails, look for any .bin files
    bin_files = glob.glob(os.path.join(search_dir, "*.bin"))
    if bin_files:
        # Sort by modification time (newest first)
        bin_files.sort(key=os.path.getmtime, reverse=True)
        latest = bin_files[0]
        logger.info(f"Found latest .bin file: {latest}")
        return latest

    logger.warning(f"No checkpoints found in {search_dir}")
    return None


def setup_model_and_tokenizer():
    """Initialize model and tokenizer with LoRA support."""
    config = {
        "max_seq_length": 4096 * 2,
        "lora_rank": 64,
        "gpu_memory_utilization": 0.6,
        "model_name": MODEL_NAME,
        "target_modules": [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    }

    logger.info(f"Setting up model {config['model_name']} with LoRA support...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config["model_name"],
        max_seq_length=config["max_seq_length"],
        load_in_4bit=True,
        fast_inference=True,
        max_lora_rank=config["lora_rank"],
        gpu_memory_utilization=config["gpu_memory_utilization"],
    )

    # Setup LoRA
    model = FastLanguageModel.get_peft_model(
        model,
        r=config["lora_rank"],
        target_modules=config["target_modules"],
        lora_alpha=config["lora_rank"],
        use_gradient_checkpointing=True,
        random_state=3407,
    )

    logger.info("Model and tokenizer setup complete.")
    return model, tokenizer


def get_sampling_params(temperature: float = 0.7, max_tokens: int = 4096) -> SamplingParams:
    """Get sampling parameters for generation."""
    return SamplingParams(
        temperature=temperature,
        top_p=0.95,
        max_tokens=max_tokens,
    )


def extract_function_calls(text: str) -> List[Dict[str, Any]]:
    """Extract function calls from a text."""
    import json
    import re

    # Pattern to match JSON objects
    pattern = r"\{(?:[^{}]|(?:\{(?:[^{}]|(?:\{[^{}]*\}))*\}))*\}"
    json_matches = re.findall(pattern, text)

    function_calls = []
    for json_str in json_matches:
        try:
            obj = json.loads(json_str)
            if "function" in obj:
                function_calls.append(obj)
        except json.JSONDecodeError:
            continue

    return function_calls


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

PLEASE CONSIDER CHAT HISTORY WHEN ANSWERING THE QUESTION.
ONLY ANSWER WHEN YOU HAVE 100% CONFIDENCE IN THE SEARCH RESULTS, ELSE CONTINUE SEARCHING.
PLEASE SEARCH MULTIPLE TIMES WITH DIFFERENT QUERIES.

You may also reason in any message, think step by step about how to answer the question. Wrap your reasoning in <think> and </think> tags.

{json.dumps(SEARCH_TOOL_DEFINITION, indent=2)}

Question: {q}
"""
    return user_prompt


def format_search_results(results: Union[str, List[str]]) -> str:
    """
    Format search results for display.

    Args:
        results: Search results as string or list of strings

    Returns:
        Formatted search results
    """
    if isinstance(results, list):
        content = "\n".join([f"Result {i + 1}:\n{r}\n------" for i, r in enumerate(results)])
    else:
        content = results

    return f"\n===== SEARCH RESULTS =====\n{content}\n===========================\n"


class DeepSearchCLI:
    """CLI for interacting with the model and search functionality."""

    def __init__(
        self,
        lora_path: Optional[str] = None,
        temperature: float = 0.7,
        system_prompt: Optional[str] = None,
    ):
        """
        Initialize the CLI.

        Args:
            lora_path: Path to LoRA weights (None for base model)
            temperature: Sampling temperature
            system_prompt: Optional system prompt to guide the model's behavior
        """
        self.model, self.tokenizer = setup_model_and_tokenizer()
        self.lora_path = lora_path
        self.temperature = temperature
        self.sampling_params = get_sampling_params(temperature)
        self.lora_request = None
        self.history = []
        self.search_history = []
        self.system_prompt = (
            system_prompt
            or f"""Cutting Knowledge Date: December 2023
Today Date: {datetime.now().strftime("%d %b %Y")}

When you receive a tool call response, use the output to format an answer to the original user question.

You are a helpful assistant with tool calling capabilities."""
        )

        # Load LoRA if specified
        if self.lora_path:
            logger.info(f"Loading LoRA adapter from {self.lora_path}...")
            self.lora_request = self.model.load_lora(self.lora_path)
            if self.lora_request:
                logger.info(f"LoRA adapter loaded successfully: {self.lora_request}")
            else:
                logger.error("Failed to load LoRA adapter")

    def generate(self, prompt: str, max_generations: int = 20) -> str:
        """
        Generate a response to the prompt using agentic mechanism.

        Args:
            prompt: The prompt to generate a response to
            max_generations: Maximum number of turns in the conversation

        Returns:
            The generated response after completing the conversation
        """
        # Initialize chat state
        chat_state = {
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": build_user_prompt(prompt)},
            ],
            "finished": False,
        }

        # Agent loop
        for i in range(max_generations):
            # Generate response
            chat_state = self._run_agent_generation(chat_state)

            # Check if conversation is finished
            chat_state = self._check_finished_chat(chat_state)
            if chat_state.get("finished"):
                break

            # Process tool calls if any
            chat_state = self._run_tool_calls(chat_state)

        # Get final response
        final_response = chat_state["messages"][-1]["content"]

        # Update history
        self.history.append({"role": "user", "content": prompt})
        self.history.append({"role": "assistant", "content": final_response})

        return final_response

    def _run_agent_generation(self, chat_state: dict) -> dict:
        """Run a single generation step for the agent."""
        formatted_prompt = self.tokenizer.apply_chat_template(
            chat_state["messages"],
            tokenize=False,
            add_generation_prompt=True,
        )

        start_time = time.time()
        if self.lora_request:
            response = self.model.fast_generate(
                [formatted_prompt],
                sampling_params=self.sampling_params,
                lora_request=self.lora_request,
            )
        else:
            response = self.model.fast_generate(
                [formatted_prompt],
                sampling_params=self.sampling_params,
            )

        gen_time = time.time() - start_time
        logger.debug(f"Generation completed in {gen_time:.2f} seconds")

        if hasattr(response[0], "outputs"):
            response_text = response[0].outputs[0].text
        else:
            response_text = response[0]

        # Extract assistant response
        assistant_response = response_text.split("<|start_header_id|>assistant<|end_header_id|>")[-1]

        chat_state["messages"].append({"role": "assistant", "content": assistant_response})

        return chat_state

    def _check_finished_chat(self, chat_state: dict) -> dict:
        """Check if the chat is finished (no more function calls)."""
        if chat_state.get("finished"):
            return chat_state

        assert chat_state["messages"][-1]["role"] == "assistant", "Expected the last role to be assistant"

        assistant_response = chat_state["messages"][-1]["content"]
        function_calls = extract_json_objects(assistant_response)

        if len(function_calls) == 0:
            chat_state["finished"] = True

        return chat_state

    def _run_tool_calls(self, chat_state: dict) -> dict:
        """Execute tool calls found in chat state."""
        if chat_state.get("finished"):
            return chat_state

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

                # Print search results to terminal
                # logger.info("\n===== SEARCH RESULTS =====")
                # logger.info(
                #     results
                # )  # The results are already formatted with Result 1:, Result 2:, etc.
                # logger.info("===========================\n")

                chat_state["messages"].append({"role": "ipython", "content": results})

                # Record search in history
                search_entry = {
                    "turn": len(self.history) // 2,
                    "searches": [{"query": query, "results": results}],
                }
                self.search_history.append(search_entry)

        except Exception as e:
            logger.error(f"Error during tool call: {str(e)}")
            chat_state["messages"].append({"role": "system", "content": f"Error during post-processing: {str(e)}"})
            chat_state["finished"] = True

        return chat_state

    def clear_history(self):
        """Clear the conversation history."""
        self.history = []
        self.search_history = []
        logger.info("Conversation history cleared.")

    def set_system_prompt(self, prompt: str):
        """
        Set a new system prompt.

        Args:
            prompt: The new system prompt
        """
        if not prompt:
            logger.warning("System prompt cannot be empty. Using default.")
            return

        self.system_prompt = prompt
        logger.info("System prompt updated.")
        logger.info(f"New system prompt: {self.system_prompt}")

    def display_welcome(self):
        """Display welcome message."""
        model_type = "LoRA" if self.lora_path else "Base"
        logger.info(f"\n{'=' * 50}")
        logger.info(f"DeepSearch CLI - {model_type} Model")
        logger.info(f"Model: {MODEL_NAME}")
        logger.info(f"Temperature: {self.temperature}")
        if self.lora_path:
            logger.info(f"LoRA Path: {self.lora_path}")
        logger.info(f"System Prompt: {self.system_prompt}")
        logger.info(f"{'=' * 50}")
        logger.info("Type 'help' to see available commands.")

    def print_pretty_chat_history(self):
        """Print the full chat history in a pretty format, including searches."""
        if not self.history:
            logger.info("No chat history available.")
            return

        logger.info("\n" + "=" * 80)
        logger.info("CHAT HISTORY WITH SEARCH DETAILS")
        logger.info("=" * 80)

        # Group history into conversation turns
        for i in range(0, len(self.history), 2):
            turn_number = i // 2

            # Print user message
            if i < len(self.history):
                user_msg = self.history[i]["content"]
                logger.info(f"\n[Turn {turn_number + 1}] USER: ")
                logger.info("-" * 40)
                logger.info(user_msg)

            # Print searches associated with this turn if any
            for search_entry in self.search_history:
                if search_entry["turn"] == turn_number:
                    for idx, search in enumerate(search_entry["searches"]):
                        logger.info(f'\n🔍 SEARCH {idx + 1}: "{search["query"]}"')
                        logger.info("-" * 40)
                        logger.info(search["results"])

            # Print assistant response
            if i + 1 < len(self.history):
                assistant_msg = self.history[i + 1]["content"]
                logger.info(f"\n[Turn {turn_number + 1}] ASSISTANT: ")
                logger.info("-" * 40)
                logger.info(assistant_msg)

        logger.info("\n" + "=" * 80 + "\n")

    def save_chat_history(self, filepath=None):
        """
        Save chat history to a file.

        Args:
            filepath: Path to save file (if None, auto-generate based on timestamp)

        Returns:
            Path to the saved file
        """
        if not self.history:
            logger.info("No chat history to save.")
            return None

        # Generate a default filepath if none provided
        if filepath is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_type = "lora" if self.lora_path else "base"
            filepath = os.path.join(OUTPUT_DIR, f"chat_history_{model_type}_{timestamp}.txt")

            # Ensure the directory exists
            os.makedirs(os.path.dirname(filepath), exist_ok=True)

        # Prepare chat history data
        pretty_history = []

        # Group history into conversation turns
        for i in range(0, len(self.history), 2):
            turn_number = i // 2
            turn_data = {
                "turn": turn_number + 1,
                "user": self.history[i]["content"] if i < len(self.history) else "",
                "searches": [],
                "assistant": self.history[i + 1]["content"] if i + 1 < len(self.history) else "",
            }

            # Add searches for this turn
            for search_entry in self.search_history:
                if search_entry["turn"] == turn_number:
                    turn_data["searches"].extend(search_entry["searches"])

            pretty_history.append(turn_data)

        # Write to file
        try:
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(f"{'=' * 80}\n")
                f.write("DEEPSEARCH CHAT HISTORY\n")
                f.write(f"Model: {MODEL_NAME}\n")
                f.write(f"LoRA Path: {self.lora_path if self.lora_path else 'None'}\n")
                f.write(f"Temperature: {self.temperature}\n")
                f.write(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"{'=' * 80}\n\n")

                for turn in pretty_history:
                    f.write(f"[Turn {turn['turn']}] USER:\n")
                    f.write(f"{'-' * 40}\n")
                    f.write(f"{turn['user']}\n\n")

                    # Write searches
                    for i, search in enumerate(turn["searches"]):
                        f.write(f'🔍 SEARCH {i + 1}: "{search["query"]}"\n')
                        f.write(f"{'-' * 40}\n")
                        f.write(f"{search['results']}\n\n")

                    f.write(f"[Turn {turn['turn']}] ASSISTANT:\n")
                    f.write(f"{'-' * 40}\n")
                    f.write(f"{turn['assistant']}\n\n")
                    f.write(f"{'=' * 40}\n\n")

            logger.info(f"Chat history saved to: {filepath}")
            return filepath

        except Exception as e:
            logger.error(f"Error saving chat history: {e}")
            return None

    def save_chat_history_json(self, filepath=None):
        """
        Save chat history to a JSON file.

        Args:
            filepath: Path to save file (if None, auto-generate based on timestamp)

        Returns:
            Path to the saved file
        """
        if not self.history:
            logger.info("No chat history to save.")
            return None

        # Generate a default filepath if none provided
        if filepath is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_type = "lora" if self.lora_path else "base"
            filepath = os.path.join(OUTPUT_DIR, f"chat_history_{model_type}_{timestamp}.json")

            # Ensure the directory exists
            os.makedirs(os.path.dirname(filepath), exist_ok=True)

        # Prepare chat history data
        history_data = {
            "model": MODEL_NAME,
            "lora_path": self.lora_path if self.lora_path else None,
            "temperature": self.temperature,
            "timestamp": datetime.now().isoformat(),
            "turns": [],
        }

        # Group history into conversation turns
        for i in range(0, len(self.history), 2):
            turn_number = i // 2
            turn_data = {
                "turn": turn_number + 1,
                "user": self.history[i]["content"] if i < len(self.history) else "",
                "searches": [],
                "assistant": self.history[i + 1]["content"] if i + 1 < len(self.history) else "",
            }

            # Add searches for this turn
            for search_entry in self.search_history:
                if search_entry["turn"] == turn_number:
                    turn_data["searches"].extend(search_entry["searches"])

            history_data["turns"].append(turn_data)

        # Write to file
        try:
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(history_data, f, indent=2, ensure_ascii=False)

            logger.info(f"Chat history saved to JSON: {filepath}")
            return filepath

        except Exception as e:
            logger.error(f"Error saving chat history to JSON: {e}")
            return None

    def display_help(self):
        """Display help information."""
        logger.info("\n===== Commands =====")
        logger.info("search <query>  - Search for information")
        logger.info("system <prompt> - Set a new system prompt")
        logger.info("clear           - Clear conversation history")
        logger.info("history         - Display full chat history with searches")
        logger.info("save            - Save chat history to a text file")
        logger.info("savejson        - Save chat history to a JSON file")
        logger.info("help            - Display this help message")
        logger.info("exit/quit       - Exit the program")
        logger.info("Any other input will be treated as a prompt to the model.")
        logger.info("===================\n")

    def run(self):
        """Run the CLI."""
        self.display_welcome()

        while True:
            try:
                user_input = input("\n> ").strip()

                if not user_input:
                    continue

                if user_input.lower() in ["exit", "quit"]:
                    logger.info("Exiting...")
                    break

                if user_input.lower() == "help":
                    self.display_help()
                    continue

                if user_input.lower() == "clear":
                    self.clear_history()
                    continue

                if user_input.lower() == "history":
                    self.print_pretty_chat_history()
                    continue

                if user_input.lower() == "save":
                    self.save_chat_history()
                    continue

                if user_input.lower() == "savejson":
                    self.save_chat_history_json()
                    continue

                if user_input.lower().startswith("system "):
                    new_prompt = user_input[7:].strip()
                    self.set_system_prompt(new_prompt)
                    continue

                if user_input.lower().startswith("search "):
                    query = user_input[7:].strip()
                    if query:
                        try:
                            results = search(query, return_type=str)
                            formatted_results = format_search_results(results)
                            logger.info(formatted_results)

                            # Add to search history
                            search_entry = {
                                "turn": len(self.history) // 2,
                                "searches": [{"query": query, "results": results}],
                            }
                            self.search_history.append(search_entry)
                        except Exception as e:
                            logger.error(f"Error searching: {e}")
                    else:
                        logger.warning("Please provide a search query.")
                    continue

                # Process as a prompt to the model
                logger.info("\nGenerating response...")
                response = self.generate(user_input)
                logger.info("\n----- Response -----")
                logger.info(response)

            except KeyboardInterrupt:
                logger.info("\nExiting...")
                break
            except Exception as e:
                logger.error(f"Error: {e}")


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


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="DeepSearch CLI")
    parser.add_argument(
        "--lora_path",
        type=str,
        default="auto",
        help="Path to LoRA weights (None for base model, 'auto' for auto-detection)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature (default: 0.7)",
    )
    parser.add_argument(
        "--system_prompt",
        type=str,
        default=None,
        help="System prompt to guide model behavior",
    )
    args = parser.parse_args()

    # Auto-detect LoRA path if requested
    lora_path = None
    if args.lora_path and args.lora_path.lower() != "none":
        if args.lora_path == "auto":
            detected_path = find_latest_checkpoint()
            if detected_path:
                lora_path = detected_path
                logger.info(f"Auto-detected LoRA path: {lora_path}")
            else:
                logger.warning("No LoRA checkpoint found. Using base model.")
        else:
            lora_path = args.lora_path

    # Initialize and run the CLI
    cli = DeepSearchCLI(
        lora_path=lora_path,
        temperature=args.temperature,
        system_prompt=args.system_prompt,
    )
    cli.run()


if __name__ == "__main__":
    # Ensure the vectorstore is loaded
    if load_vectorstore() is None:
        logger.warning("FAISS vectorstore could not be loaded. Search functionality may not work.")

    main()
