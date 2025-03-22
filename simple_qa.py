#!/usr/bin/env python3
"""
Simple command-line Q&A environment for testing with search functionality.
"""

import asyncio
import json
import random
import sys
import time
from typing import Any, Dict

# Import our search module (ensure these functions follow the new interfaces)
from search_module import get_question_answer, get_question_count, search


class SimpleQAEnvironment:
    """Simple command-line environment for Q&A with search capability."""

    def __init__(self):
        self.score = {"correct": 0, "incorrect": 0, "total": 0}
        self.session_data = []
        self.current_question = None

    def display_welcome(self):
        """Display welcome message and instructions."""
        print("\n===== Search & Answer Environment =====")
        print("Answer questions using the search tool to find relevant information.")
        print("Type 'q' to quit, 'h' for help.\n")

    def display_help(self):
        """Display help information."""
        print("\n===== Commands =====")
        print("n          - Get a new question")
        print("s <query>  - Search for information (e.g., s program launch date)")
        print("a <answer> - Submit your answer")
        print("h          - Display this help message")
        print("q          - Quit the program\n")

    def display_question(self, question: str):
        """Display the current question."""
        print("\n===== QUESTION =====")
        print(question)
        print("=====================\n")

    def get_new_question(self) -> str:
        """Get a new random question and set it as current."""
        total_questions = get_question_count()
        question_id = random.randint(0, total_questions - 1)

        # Updated to match new interface: get_question_answer now returns a dict.
        qa = get_question_answer(question_id)
        question = qa["question"]
        correct_answer = qa["answer"]

        question_data = {
            "id": question_id,
            "question": question,
            "correct_answer": correct_answer,
            "start_time": time.time(),
            "searches": [],
        }
        self.current_question = question_data
        return question

    def perform_search(self, query: str):
        """Perform a search with the given query."""
        if not query:
            print("Please provide a search query.")
            return

        try:
            print("\n===== SEARCH RESULTS =====")
            results = search(query)
            print(results)
            print("==========================\n")

            # Record search in current question data if available.
            if self.current_question is not None:
                self.current_question["searches"].append(query)

        except Exception as e:
            print(f"Error searching: {str(e)}")

    async def process_answer(self, user_answer: str):
        """Process and verify the user's answer."""
        if self.current_question is None:
            print("Please get a question first.")
            return

        if not user_answer:
            print("Please provide an answer.")
            return

        # Record answer and calculate time taken.
        self.current_question["user_answer"] = user_answer
        self.current_question["end_time"] = time.time()
        self.current_question["time_taken"] = (
            self.current_question["end_time"] - self.current_question["start_time"]
        )

        try:
            print("\nVerifying your answer...")
            correct = await verify(
                user_answer,
                self.current_question["question"],
                self.current_question["correct_answer"],
                router,
            )

            # Update score and inform the user.
            self.score["total"] += 1
            if correct:
                self.score["correct"] += 1
                print("\n✓ Your answer is CORRECT!")
            else:
                self.score["incorrect"] += 1
                print("\n✗ Your answer is INCORRECT.")
                print(
                    f"\nThe correct answer is:\n{self.current_question['correct_answer']}"
                )

            print(f"\nScore: {self.score['correct']}/{self.score['total']}")

            # Record the result and add the current question to the session data.
            self.current_question["is_correct"] = correct
            self.session_data.append(self.current_question)

            # Clear the current question.
            self.current_question = None

        except Exception as e:
            print(f"Error verifying answer: {str(e)}")

    def save_session(self):
        """Save the session data to a file."""
        if not self.session_data:
            return

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"qa_session_{timestamp}.json"

        session_data = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "score": self.score,
            "questions": self.session_data,
        }

        try:
            with open(filename, "w") as f:
                json.dump(session_data, f, indent=2)
            print(f"\nSession data saved to {filename}")
        except Exception as e:
            print(f"Error saving session data: {str(e)}")

    async def run(self):
        """Run the main command loop."""
        self.display_welcome()

        while True:
            command = input("\n> ").strip()

            if not command:
                continue

            # Process commands.
            if command.lower() == "q":
                break
            elif command.lower() == "h":
                self.display_help()
            elif command.lower() == "n":
                question = self.get_new_question()
                self.display_question(question)
            elif command.lower().startswith("s "):
                query = command[2:].strip()
                self.perform_search(query)
            elif command.lower().startswith("a "):
                answer = command[2:].strip()
                await self.process_answer(answer)
            else:
                print("Unknown command. Type 'h' for help.")

        # Save session data on exit.
        self.save_session()
        print("\nThank you for using the Q&A environment!")


async def main():
    """Main function to start the application."""
    env = SimpleQAEnvironment()
    await env.run()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nProgram terminated by user.")
    except Exception as e:
        print(f"\nError: {str(e)}")
