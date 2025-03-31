"""
Simple CLI inference script with search functionality.

This script allows interaction with the merged 16-bit model
and provides search functionality for data retrieval.
"""

import argparse
import json
import os
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import SamplingParams

from src.search_module import load_vectorstore, search


def setup_model_and_tokenizer(model_path: str):
    """Initialize model and tokenizer."""
    print(f"Setting up model from {model_path}...")

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype="float16",
        device_map="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    print("Model and tokenizer setup complete.")
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
        model_path: str,
        temperature: float = 0.7,
        system_prompt: Optional[str] = None,
    ):
        """
        Initialize the CLI.

        Args:
            model_path: Path to the merged 16-bit model
            temperature: Sampling temperature
            system_prompt: Optional system prompt to guide the model's behavior
        """
        self.model, self.tokenizer = setup_model_and_tokenizer(model_path)
        self.temperature = temperature
        self.sampling_params = get_sampling_params(temperature)
        self.history = []
        self.search_history = []
        self.system_prompt = (
            system_prompt
            or f"""Cutting Knowledge Date: December 2023
Today Date: {datetime.now().strftime("%d %b %Y")}

When you receive a tool call response, use the output to format an answer to the original user question.

You are a helpful assistant with tool calling capabilities."""
        )

    def _run_agent_generation(self, chat_state: dict) -> dict:
        """Run a single generation step for the agent."""
        formatted_prompt = self.tokenizer.apply_chat_template(
            chat_state["messages"],
            tokenize=False,
            add_generation_prompt=True,
        )

        start_time = time.time()
        inputs = self.tokenizer(formatted_prompt, return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=self.sampling_params.max_tokens,
            temperature=self.sampling_params.temperature,
            top_p=self.sampling_params.top_p,
            do_sample=True,
        )
        response_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        gen_time = time.time() - start_time
        print(f"Generation completed in {gen_time:.2f} seconds")

        # Extract assistant response
        assistant_response = response_text.split("<|start_header_id|>assistant<|end_header_id|>")[-1]

        chat_state["messages"].append({"role": "assistant", "content": assistant_response})

        return chat_state

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

    def _check_finished_chat(self, chat_state: dict) -> dict:
        """Check if the chat is finished (no more function calls)."""
        if chat_state.get("finished"):
            return chat_state

        assert chat_state["messages"][-1]["role"] == "assistant", "Expected the last role to be assistant"

        assistant_response = chat_state["messages"][-1]["content"]
        function_calls = extract_function_calls(assistant_response)

        if len(function_calls) == 0:
            chat_state["finished"] = True

        return chat_state

    def _run_tool_calls(self, chat_state: dict) -> dict:
        """Execute tool calls found in chat state."""
        if chat_state.get("finished"):
            return chat_state

        try:
            assistant_response = chat_state["messages"][-1]["content"]
            function_calls = extract_function_calls(assistant_response)

            if len(function_calls) > 1:
                print("Multiple function calls found in assistant response")
                raise ValueError("Expected only one function call in assistant response")

            elif len(function_calls) == 1:
                function_call = function_calls[0]
                query = function_call["function"]["parameters"]["query"]
                print(f"🔍 Search Query: {query}")

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
            print(f"Error during tool call: {str(e)}")
            chat_state["messages"].append({"role": "system", "content": f"Error during post-processing: {str(e)}"})
            chat_state["finished"] = True

        return chat_state

    def clear_history(self):
        """Clear the conversation history."""
        self.history = []
        self.search_history = []
        print("Conversation history cleared.")

    def set_system_prompt(self, prompt: str):
        """
        Set a new system prompt.

        Args:
            prompt: The new system prompt
        """
        if not prompt:
            print("System prompt cannot be empty. Using default.")
            return

        self.system_prompt = prompt
        print("System prompt updated.")
        print(f"New system prompt: {self.system_prompt}")

    def display_welcome(self):
        """Display welcome message."""
        print(f"\n{'=' * 50}")
        print(f"DeepSearch CLI - {self.model.name_or_path}")
        print(f"Model: {self.model.name_or_path}")
        print(f"Temperature: {self.temperature}")
        print(f"System Prompt: {self.system_prompt}")
        print(f"{'=' * 50}")
        print("Type 'help' to see available commands.")

    def print_pretty_chat_history(self):
        """Print the full chat history in a pretty format, including searches."""
        if not self.history:
            print("No chat history available.")
            return

        print("\n" + "=" * 80)
        print("CHAT HISTORY WITH SEARCH DETAILS")
        print("=" * 80)

        # Group history into conversation turns
        for i in range(0, len(self.history), 2):
            turn_number = i // 2

            # Print user message
            if i < len(self.history):
                user_msg = self.history[i]["content"]
                print(f"\n[Turn {turn_number + 1}] USER: ")
                print("-" * 40)
                print(user_msg)

            # Print searches associated with this turn if any
            for search_entry in self.search_history:
                if search_entry["turn"] == turn_number:
                    for idx, search in enumerate(search_entry["searches"]):
                        print(f'\n🔍 SEARCH {idx + 1}: "{search["query"]}"')
                        print("-" * 40)
                        print(search["results"])

            # Print assistant response
            if i + 1 < len(self.history):
                assistant_msg = self.history[i + 1]["content"]
                print(f"\n[Turn {turn_number + 1}] ASSISTANT: ")
                print("-" * 40)
                print(assistant_msg)

        print("\n" + "=" * 80 + "\n")

    def save_chat_history(self, filepath=None):
        """
        Save chat history to a file.

        Args:
            filepath: Path to save file (if None, auto-generate based on timestamp)

        Returns:
            Path to the saved file
        """
        if not self.history:
            print("No chat history to save.")
            return None

        # Generate a default filepath if none provided
        if filepath is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = os.path.join(os.getcwd(), f"chat_history_{timestamp}.txt")

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
                f.write(f"Model: {self.model.name_or_path}\n")
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

            print(f"Chat history saved to: {filepath}")
            return filepath

        except Exception as e:
            print(f"Error saving chat history: {e}")
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
            print("No chat history to save.")
            return None

        # Generate a default filepath if none provided
        if filepath is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = os.path.join(os.getcwd(), f"chat_history_{timestamp}.json")

            # Ensure the directory exists
            os.makedirs(os.path.dirname(filepath), exist_ok=True)

        # Prepare chat history data
        history_data = {
            "model": self.model.name_or_path,
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

            print(f"Chat history saved to JSON: {filepath}")
            return filepath

        except Exception as e:
            print(f"Error saving chat history to JSON: {e}")
            return None

    def display_help(self):
        """Display help information."""
        print("\n===== Commands =====")
        print("search <query>  - Search for information")
        print("system <prompt> - Set a new system prompt")
        print("clear           - Clear conversation history")
        print("history         - Display full chat history with searches")
        print("save            - Save chat history to a text file")
        print("savejson        - Save chat history to a JSON file")
        print("help            - Display this help message")
        print("exit/quit       - Exit the program")
        print("Any other input will be treated as a prompt to the model.")
        print("===================\n")

    def run(self):
        """Run the CLI."""
        self.display_welcome()

        while True:
            try:
                user_input = input("\n> ").strip()

                if not user_input:
                    continue

                if user_input.lower() in ["exit", "quit"]:
                    print("Exiting...")
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
                            print(formatted_results)

                            # Add to search history
                            search_entry = {
                                "turn": len(self.history) // 2,
                                "searches": [{"query": query, "results": results}],
                            }
                            self.search_history.append(search_entry)
                        except Exception as e:
                            print(f"Error searching: {e}")
                    else:
                        print("Please provide a search query.")
                    continue

                # Process as a prompt to the model
                print("\nGenerating response...")
                response = self.generate(user_input)
                print("\n----- Response -----")
                print(response)

            except KeyboardInterrupt:
                print("\nExiting...")
                break
            except Exception as e:
                print(f"Error: {e}")


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
        "--model_path",
        type=str,
        default="trainer_output_example/model_merged_16bit",
        help="Path to the merged 16-bit model (default: trainer_output_example/model_merged_16bit)",
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

    # Initialize and run the CLI
    cli = DeepSearchCLI(
        model_path=args.model_path,
        temperature=args.temperature,
        system_prompt=args.system_prompt,
    )
    cli.run()


if __name__ == "__main__":
    # Ensure the vectorstore is loaded
    if load_vectorstore() is None:
        print("FAISS vectorstore could not be loaded. Search functionality may not work.")

    main()
