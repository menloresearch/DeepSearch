"""
Simple CLI inference script with search functionality.

This script allows interaction with the merged 16-bit model
and provides search functionality for data retrieval.
"""

import argparse
import os
import time
from datetime import datetime

from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import SamplingParams

from src import (
    apply_chat_template,
    build_user_prompt,
    extract_search_query,
    format_search_results,
    get_system_prompt,
)
from src.deepsearch.search_module import load_vectorstore, search


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


class DeepSearchCLI:
    """CLI for interacting with the model and search functionality."""

    def __init__(
        self,
        model_path: str,
        temperature: float = 0.7,
        system_prompt: str | None = None,
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
        self.system_prompt = system_prompt or get_system_prompt()

    def _run_agent_generation(self, chat_state: dict) -> dict:
        """Run a single generation step for the agent."""
        # Format the chat state using the same template as training
        formatted_prompt = apply_chat_template(chat_state, tokenizer=self.tokenizer)["text"]

        start_time = time.time()
        inputs = self.tokenizer(formatted_prompt, return_tensors="pt", add_special_tokens=False).to(self.model.device)
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
        # Initialize chat state with the same structure as training
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
        """Check if the chat is finished (no more search queries)."""
        if chat_state.get("finished"):
            return chat_state

        assert chat_state["messages"][-1]["role"] == "assistant", "Expected the last role to be assistant"

        assistant_response = chat_state["messages"][-1]["content"]
        search_query = extract_search_query(assistant_response)

        if not search_query:
            chat_state["finished"] = True

        return chat_state

    def _run_tool_calls(self, chat_state: dict) -> dict:
        """Execute tool calls found in chat state."""
        if chat_state.get("finished"):
            return chat_state

        try:
            assistant_response = chat_state["messages"][-1]["content"]
            search_query = extract_search_query(assistant_response)

            if search_query:
                print(f"🔍 Search Query: {search_query}")

                results = search(search_query, return_type=str, results=2)
                # Wrap results in <information> tags
                formatted_results = f"<information>{results}</information>"

                # Print search results to terminal
                print("\n===== SEARCH RESULTS =====")
                print(results)
                print("===========================\n")

                chat_state["messages"].append({"role": "ipython", "content": formatted_results})

                # Record search in history
                search_entry = {
                    "turn": len(self.history) // 2,
                    "searches": [{"query": search_query, "results": results}],
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

    def display_help(self):
        """Display help information."""
        print("\n===== Commands =====")
        print("search <query>  - Search for information")
        print("system <prompt> - Set a new system prompt")
        print("clear           - Clear conversation history")
        print("history         - Display full chat history with searches")
        print("save            - Save chat history to a text file")
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
