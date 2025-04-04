"""
Evaluation utilities for RL training.
"""

import inspect
from datetime import datetime

from src.agent import Agent
from src.config import logger
from src.search_module import get_qa_dataset
from src.tokenizer_adapter import LlamaTokenizerAdapter, R1DistilTokenizerAdapter


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


def run_eval(generate_fn, verify_fn, tokenizer, max_generations=20, output_file=None, debug_file=None):
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

    # Create agent with appropriate adapter based on tokenizer
    tokenizer_name = tokenizer.name_or_path.lower()
    if "deepseek-ai/deepseek-r1-distill" in tokenizer_name:
        adapter = R1DistilTokenizerAdapter()
    elif "llama" in tokenizer_name:
        adapter = LlamaTokenizerAdapter()
    else:
        adapter = R1DistilTokenizerAdapter()

    agent = Agent(adapter)
    agentic_outputs = agent.run_agent(generate_fn, tokenizer, questions, max_generations)
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
