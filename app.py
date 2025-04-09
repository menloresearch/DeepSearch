"""
Gradio web interface for DeepSearch.

This module provides a simple web interface for interacting with the DeepSearch model
using Gradio. It implements the core functionality directly for better modularity.
"""

import re
import sys
import time
from typing import Iterator, cast

import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizer

# Import from config
from config import GENERATOR_MODEL_DIR, logger
from src import (
    apply_chat_template,
    build_user_prompt,
    format_search_results,
    get_system_prompt,
)
from src.search_module import load_vectorstore, search

# TODO: check if can reuse tokenizer adapter


def extract_answer_tag(text: str) -> tuple[bool, str | None]:
    """Check if text contains an answer tag and extract the answer content if found.

    Returns:
        tuple: (has_answer, answer_content)
    """
    pattern = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)
    match = re.search(pattern, text)
    if match:
        content = match.group(1).strip()
        return True, content
    return False, None


def extract_thinking_content(text: str) -> str | None:
    """Extract thinking content from text between <think> tags."""
    pattern = re.compile(r"<think>(.*?)</think>", re.DOTALL | re.IGNORECASE)
    match = re.search(pattern, text)
    if match:
        content = match.group(1).strip()
        return content
    return None


def extract_search_query(text: str) -> str | None:
    """Extract search query from text between <search> tags (Simplified)."""
    pattern = re.compile(r"<search>(.*?)</search>", re.DOTALL | re.IGNORECASE)
    match = re.search(pattern, text)
    if match:
        content = match.group(1).strip()
        return content
    return None


def setup_model_and_tokenizer(model_path: str):
    """Initialize model and tokenizer."""
    logger.info(f"Setting up model from {model_path}...")

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype="float16",
        device_map="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    # Defaulting to the one used in inference.py, adjust if needed for your specific model
    assistant_marker = "<|start_header_id|>assistant<|end_header_id|>"
    logger.info(f"Using assistant marker: '{assistant_marker}' for response splitting.")

    logger.info("Model and tokenizer setup complete.")
    return model, tokenizer, assistant_marker


def get_sampling_params(temperature: float = 0.7, max_tokens: int = 4096):
    """Get sampling parameters for generation."""
    return {
        "temperature": temperature,
        "top_p": 0.95,
        "max_new_tokens": max_tokens,
        "do_sample": True,
    }


def create_interface(model_path: str, temperature: float = 0.7, system_prompt: str | None = None):
    """Create Gradio interface for DeepSearch."""
    model, tokenizer, assistant_marker = setup_model_and_tokenizer(model_path)
    system_prompt = system_prompt or get_system_prompt()
    tokenizer_for_template = cast(PreTrainedTokenizer, tokenizer)

    def get_chat_num_tokens(current_chat_state: dict) -> int:
        """Helper to get number of tokens in chat state."""
        try:
            chat_text = apply_chat_template(current_chat_state, tokenizer=tokenizer_for_template)["text"]
            input_ids = tokenizer.encode(chat_text, add_special_tokens=False)
            return len(input_ids)
        except Exception as e:
            logger.error(f"Error calculating token count: {e}")
            return sys.maxsize

    def stream_agent_response(
        message: str,
        history_gr: list[gr.ChatMessage],
        temp: float,
        max_iter: int = 20,
        num_search_results: int = 2,
    ) -> Iterator[list[gr.ChatMessage]]:
        """Stream agent responses following agent.py/inference.py logic."""
        chat_state = {
            "messages": [{"role": "system", "content": system_prompt}],
            "finished": False,
        }
        # Convert Gradio history to internal format, skip last user msg (passed separately)
        processed_history = history_gr[:-1] if history_gr else []
        for msg_obj in processed_history:
            role = getattr(msg_obj, "role", "unknown")
            content = getattr(msg_obj, "content", "")
            if role == "user":
                chat_state["messages"].append({"role": "user", "content": build_user_prompt(content)})
            elif role == "assistant":
                chat_state["messages"].append({"role": "assistant", "content": content})

        chat_state["messages"].append({"role": "user", "content": build_user_prompt(message)})

        initial_token_length = get_chat_num_tokens(chat_state)
        max_new_tokens_allowed = get_sampling_params(temp)["max_new_tokens"]

        messages = history_gr

        start_time = time.time()
        iterations = 0
        last_assistant_response = ""

        while not chat_state.get("finished", False) and iterations < max_iter:
            iterations += 1
            current_turn_start_time = time.time()

            think_msg_idx = len(messages)
            messages.append(
                gr.ChatMessage(
                    role="assistant",
                    content="Thinking...",
                    metadata={"title": "🧠 Thinking", "status": "pending"},
                )
            )
            yield messages

            current_length_before_gen = get_chat_num_tokens(chat_state)
            if current_length_before_gen - initial_token_length > max_new_tokens_allowed:
                logger.warning(
                    f"TOKEN LIMIT EXCEEDED (Before Generation): Current {current_length_before_gen}, Start {initial_token_length}"
                )
                chat_state["finished"] = True
                messages[think_msg_idx] = gr.ChatMessage(
                    role="assistant",
                    content="Context length limit reached.",
                    metadata={"title": "⚠️ Token Limit", "status": "done"},
                )
                yield messages
                break

            try:
                generation_params = get_sampling_params(temp)
                formatted_prompt = apply_chat_template(chat_state, tokenizer=tokenizer_for_template)["text"]
                inputs = tokenizer(formatted_prompt, return_tensors="pt", add_special_tokens=False).to(model.device)

                outputs = model.generate(**inputs, **generation_params)
                full_response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

                if assistant_marker in full_response_text:
                    assistant_response = full_response_text.split(assistant_marker)[-1].strip()
                else:
                    inputs_dict = cast(dict, inputs)
                    input_token_length = len(inputs_dict["input_ids"][0])
                    assistant_response = tokenizer.decode(
                        outputs[0][input_token_length:], skip_special_tokens=True
                    ).strip()
                    logger.warning(
                        f"Assistant marker '{assistant_marker}' not found in response. Extracted via token slicing fallback."
                    )

                last_assistant_response = assistant_response
                thinking_content = extract_thinking_content(assistant_response)

                gen_time = time.time() - current_turn_start_time

                display_thinking = thinking_content if thinking_content else "Processing..."
                messages[think_msg_idx] = gr.ChatMessage(
                    role="assistant",
                    content=display_thinking,
                    metadata={"title": "🧠 Thinking", "status": "done", "duration": gen_time},
                )
                yield messages

            except Exception as e:
                logger.error(f"Error during generation: {e}")
                chat_state["finished"] = True
                messages[think_msg_idx] = gr.ChatMessage(
                    role="assistant",
                    content=f"Error during generation: {e}",
                    metadata={"title": "❌ Generation Error", "status": "done"},
                )
                yield messages
                break

            chat_state["messages"].append({"role": "assistant", "content": assistant_response})

            search_query = extract_search_query(assistant_response)

            if not search_query:
                chat_state["finished"] = True
            else:
                search_msg_idx = len(messages)
                messages.append(
                    gr.ChatMessage(
                        role="assistant",
                        content=f"Searching for: {search_query}",
                        metadata={"title": "🔍 Search", "status": "pending"},
                    )
                )
                yield messages
                search_start = time.time()
                try:
                    results = search(search_query, return_type=str, results=num_search_results)
                    search_duration = time.time() - search_start

                    messages[search_msg_idx] = gr.ChatMessage(
                        role="assistant",
                        content=f"{search_query}",
                        metadata={"title": "🔍 Search", "duration": search_duration},
                    )
                    yield messages
                    display_results = format_search_results(results)
                    messages.append(
                        gr.ChatMessage(
                            role="assistant",
                            content=display_results,
                            metadata={"title": "ℹ️ Information", "status": "done"},
                        )
                    )
                    yield messages

                    formatted_results = f"<information>{results}</information>"
                    chat_state["messages"].append({"role": "user", "content": formatted_results})

                except Exception as e:
                    search_duration = time.time() - search_start
                    logger.error(f"Search failed: {str(e)}")
                    messages[search_msg_idx] = gr.ChatMessage(
                        role="assistant",
                        content=f"Search failed: {str(e)}",
                        metadata={"title": "❌ Search Error", "status": "done", "duration": search_duration},
                    )
                    yield messages
                    chat_state["messages"].append({"role": "system", "content": f"Error during search: {str(e)}"})
                    chat_state["finished"] = True

            current_length_after_iter = get_chat_num_tokens(chat_state)
            if current_length_after_iter - initial_token_length > max_new_tokens_allowed:
                logger.warning(
                    f"TOKEN LIMIT EXCEEDED (After Iteration): Current {current_length_after_iter}, Start {initial_token_length}"
                )
                chat_state["finished"] = True
                if messages[-1].metadata.get("title") != "⚠️ Token Limit":
                    messages.append(
                        gr.ChatMessage(
                            role="assistant",
                            content="Context length limit reached during processing.",
                            metadata={"title": "⚠️ Token Limit", "status": "done"},
                        )
                    )
                    yield messages

        total_time = time.time() - start_time

        if not chat_state.get("finished", False) and iterations >= max_iter:
            logger.warning(f"Reached maximum iterations ({max_iter}) without finishing")
            messages.append(
                gr.ChatMessage(
                    role="assistant",
                    content=f"Reached maximum iterations ({max_iter}). Displaying last response:\n\n{last_assistant_response}",
                    metadata={"title": "⚠️ Max Iterations", "status": "done", "duration": total_time},
                )
            )
            yield messages
        elif chat_state.get("finished", False) and last_assistant_response:
            has_answer, answer_content = extract_answer_tag(last_assistant_response)

            if has_answer and answer_content is not None:
                display_title = "📝 Final Answer"
                display_content = answer_content
            else:
                display_title = "💡 Answer"
                display_content = last_assistant_response

            if len(messages) > 0 and messages[-1].content != display_content:
                messages.append(
                    gr.ChatMessage(
                        role="assistant",
                        content=display_content,
                        metadata={"title": display_title, "duration": total_time},
                    )
                )
                yield messages
            elif len(messages) == 0:
                messages.append(
                    gr.ChatMessage(
                        role="assistant",
                        content=display_content,
                        metadata={"title": display_title, "duration": total_time},
                    )
                )
                yield messages
            else:
                messages[-1].metadata["title"] = display_title
                messages[-1].metadata["status"] = "done"
                messages[-1].metadata["duration"] = total_time

        logger.info(f"Processing finished in {total_time:.2f} seconds.")

    with gr.Blocks(title="DeepSearch - Visible Thinking") as interface:
        gr.Markdown("# 🧠 DeepSearch with Visible Thinking")
        gr.Markdown("Watch as the AI thinks, searches, and processes information to answer your questions.")

        with gr.Row():
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(
                    [],
                    elem_id="chatbot",
                    type="messages",
                    height=600,
                    show_label=False,
                    render_markdown=True,
                    bubble_full_width=False,
                )
                msg = gr.Textbox(
                    placeholder="Type your message here...", show_label=False, container=False, elem_id="msg-input"
                )

                example_questions = [
                    "What year was the document approved by the Mission Evaluation Team?",
                    "Summarize the key findings regarding the oxygen tank failure.",
                    "Who was the commander of Apollo 13?",
                    "What were the main recommendations from the review board?",
                ]
                gr.Examples(examples=example_questions, inputs=msg, label="Example Questions", examples_per_page=4)

                with gr.Row():
                    clear = gr.Button("Clear Chat")
                    submit = gr.Button("Submit", variant="primary")

            with gr.Column(scale=1):
                gr.Markdown("### Settings")
                temp_slider = gr.Slider(minimum=0.1, maximum=1.0, value=temperature, step=0.1, label="Temperature")
                system_prompt_input = gr.Textbox(
                    label="System Prompt", value=system_prompt, lines=3, info="Controls how the AI behaves"
                )
                max_iter_slider = gr.Slider(
                    minimum=1,
                    maximum=20,
                    value=20,
                    step=1,
                    label="Max Search Iterations",
                    info="Maximum number of search-think cycles",
                )
                num_results_slider = gr.Slider(
                    minimum=1,
                    maximum=5,
                    value=2,
                    step=1,
                    label="Number of Search Results",
                    info="How many results to retrieve per search query",
                )

        def add_user_message(user_msg_text: str, history: list[gr.ChatMessage]) -> tuple[str, list[gr.ChatMessage]]:
            """Appends user message to chat history and clears input."""
            if user_msg_text and user_msg_text.strip():
                history.append(gr.ChatMessage(role="user", content=user_msg_text.strip()))
            return "", history

        submitted_msg_state = gr.State("")

        # Chain events:
        # 1. User submits -> store msg text in state
        # 2. .then() -> add_user_message (updates chatbot UI history, clears textbox)
        # 3. .then() -> stream_agent_response (takes stored msg text and updated chatbot history)
        submit.click(
            lambda msg_text: msg_text,
            inputs=[msg],
            outputs=[submitted_msg_state],
            queue=False,
        ).then(
            add_user_message,
            inputs=[submitted_msg_state, chatbot],
            outputs=[msg, chatbot],
            queue=False,
        ).then(
            stream_agent_response,
            inputs=[submitted_msg_state, chatbot, temp_slider, max_iter_slider, num_results_slider],
            outputs=chatbot,
        )

        msg.submit(
            lambda msg_text: msg_text,
            inputs=[msg],
            outputs=[submitted_msg_state],
            queue=False,
        ).then(
            add_user_message,
            inputs=[submitted_msg_state, chatbot],
            outputs=[msg, chatbot],
            queue=False,
        ).then(
            stream_agent_response,
            inputs=[submitted_msg_state, chatbot, temp_slider, max_iter_slider, num_results_slider],
            outputs=chatbot,
        )

        clear.click(lambda: ([], ""), None, [chatbot, submitted_msg_state])

        system_prompt_state = gr.State(system_prompt)
        # TODO: Currently, changing the system prompt mid-chat won't affect the ongoing stream_agent_response.
        system_prompt_input.change(lambda prompt: prompt, inputs=[system_prompt_input], outputs=[system_prompt_state])

    return interface


def main():
    """Run the Gradio app."""
    model_path = str(GENERATOR_MODEL_DIR)
    logger.info(f"Using model from config: {model_path}")

    interface = create_interface(model_path)
    interface.launch(share=True)


if __name__ == "__main__":
    if load_vectorstore() is None:
        logger.warning("⚠️ FAISS vectorstore could not be loaded. Search functionality may be unavailable.")

    main()
