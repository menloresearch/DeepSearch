"""
Gradio web interface for ReZero.

This module provides a simple web interface for interacting with the ReZero model
using Gradio. It implements the core functionality directly for better modularity.
"""

import os
import re
import sys
import time
from typing import Iterator, cast

import gradio as gr
from tavily import TavilyClient
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizer

# Import from config
from config import GENERATOR_MODEL_DIR, logger
from src import (
    apply_chat_template,
    build_user_prompt,
    format_search_results,
    get_system_prompt,
)
from src.search_module import get_qa_dataset, load_vectorstore, search


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


# Define token counting globally, needs tokenizer_for_template accessible
# Note: This requires tokenizer_for_template to be defined before this is called
# We will define tokenizer_for_template globally after model loading in main()
_tokenizer_for_template_global = None  # Placeholder


def get_chat_num_tokens(current_chat_state: dict, tokenizer: PreTrainedTokenizer) -> int:
    """Helper to get number of tokens in chat state."""
    try:
        chat_text = apply_chat_template(current_chat_state, tokenizer=tokenizer)["text"]
        # Use the passed tokenizer for encoding
        input_ids = tokenizer.encode(chat_text, add_special_tokens=False)
        return len(input_ids)
    except Exception as e:
        logger.error(f"Error calculating token count: {e}")
        return sys.maxsize


def create_deepsearch_tab(model, tokenizer, assistant_marker, system_prompt, temperature):
    """Creates the UI components and logic for the ReZero (Vector DB) tab."""
    logger.info("Creating ReZero Tab")
    # tokenizer_for_template = cast(PreTrainedTokenizer, tokenizer) # Global now

    # Load QA dataset for examples and gold answers
    try:
        _, test_dataset = get_qa_dataset()
        qa_map = {q: a for q, a in zip(test_dataset["prompt"], test_dataset["answer"])}
        example_questions = list(qa_map.keys())
        logger.info(f"Loaded {len(example_questions)} QA examples for ReZero tab.")
    except Exception as e:
        logger.error(f"Failed to load QA dataset for ReZero tab: {e}")
        qa_map = {}
        example_questions = [
            "What year was the document approved by the Mission Evaluation Team?",
            "Failed to load dataset examples.",
        ]

    # --- Agent Streaming Logic for ReZero ---
    def stream_agent_response(
        message: str,
        history_gr: list[gr.ChatMessage],
        temp: float,
        max_iter: int = 20,
        num_search_results: int = 2,
        gold_answer_state: str | None = None,
    ) -> Iterator[list[gr.ChatMessage]]:
        """Stream agent responses following agent.py/inference.py logic."""
        # Pass the globally defined (and typed) tokenizer to this scope
        local_tokenizer_for_template = _tokenizer_for_template_global
        assert local_tokenizer_for_template is not None  # Ensure it's loaded

        chat_state = {
            "messages": [{"role": "system", "content": system_prompt}],
            "finished": False,
        }
        processed_history = history_gr[:-1] if history_gr else []
        for msg_obj in processed_history:
            role = getattr(msg_obj, "role", "unknown")
            content = getattr(msg_obj, "content", "")
            if role == "user":
                chat_state["messages"].append({"role": "user", "content": build_user_prompt(content)})
            elif role == "assistant":
                chat_state["messages"].append({"role": "assistant", "content": content})

        chat_state["messages"].append({"role": "user", "content": build_user_prompt(message)})

        initial_token_length = get_chat_num_tokens(chat_state, local_tokenizer_for_template)  # Pass tokenizer
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

            current_length_before_gen = get_chat_num_tokens(chat_state, local_tokenizer_for_template)  # Pass tokenizer
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
                formatted_prompt = apply_chat_template(chat_state, tokenizer=local_tokenizer_for_template)[
                    "text"
                ]  # Use local typed tokenizer
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
                        metadata={"title": "🔍 ReZero Query", "status": "pending"},
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
                        metadata={"title": "🔍 ReZero Query", "duration": search_duration},
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

            current_length_after_iter = get_chat_num_tokens(chat_state, local_tokenizer_for_template)  # Pass tokenizer
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

        if gold_answer_state:
            logger.info("Displaying gold answer.")
            messages.append(
                gr.ChatMessage(
                    role="assistant",
                    content=gold_answer_state,
                    metadata={"title": "✅ Correct Answer (For comparison)"},
                )
            )
            yield messages
        else:
            logger.info("No gold answer to display for this query.")

    # --- UI Layout for ReZero Tab ---
    with gr.Blocks(analytics_enabled=False) as deepsearch_tab:
        gr.Markdown("# 🧠 ReZero: Enhancing LLM search ability by trying  one-more-time")
        gr.Markdown("Ask questions answered using the local vector database.")

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

                gr.Examples(
                    examples=example_questions,
                    inputs=msg,
                    label="Example Questions with correct answer for comparison",
                    examples_per_page=4,
                )

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

        # --- Event Handlers for ReZero Tab ---
        def add_user_message(user_msg_text: str, history: list[gr.ChatMessage]) -> tuple[str, list[gr.ChatMessage]]:
            """Appends user message to chat history and clears input."""
            if user_msg_text and user_msg_text.strip():
                history.append(gr.ChatMessage(role="user", content=user_msg_text.strip()))
            return "", history

        submitted_msg_state = gr.State("")
        gold_answer_state = gr.State(None)

        def check_if_example_and_store_answer(msg_text):
            gold_answer = qa_map.get(msg_text)
            logger.info(f"Checking for gold answer for: '{msg_text[:50]}...'. Found: {bool(gold_answer)}")
            return gold_answer

        submit.click(
            lambda msg_text: msg_text,
            inputs=[msg],
            outputs=[submitted_msg_state],
            queue=False,
        ).then(
            check_if_example_and_store_answer,
            inputs=[submitted_msg_state],
            outputs=[gold_answer_state],
            queue=False,
        ).then(
            add_user_message,
            inputs=[submitted_msg_state, chatbot],
            outputs=[msg, chatbot],
            queue=False,
        ).then(
            stream_agent_response,  # References the function defined within this scope
            inputs=[submitted_msg_state, chatbot, temp_slider, max_iter_slider, num_results_slider, gold_answer_state],
            outputs=chatbot,
        )

        msg.submit(
            lambda msg_text: msg_text,
            inputs=[msg],
            outputs=[submitted_msg_state],
            queue=False,
        ).then(
            check_if_example_and_store_answer,
            inputs=[submitted_msg_state],
            outputs=[gold_answer_state],
            queue=False,
        ).then(
            add_user_message,
            inputs=[submitted_msg_state, chatbot],
            outputs=[msg, chatbot],
            queue=False,
        ).then(
            stream_agent_response,  # References the function defined within this scope
            inputs=[submitted_msg_state, chatbot, temp_slider, max_iter_slider, num_results_slider, gold_answer_state],
            outputs=chatbot,
        )

        clear.click(lambda: ([], None), None, [chatbot, gold_answer_state])

        system_prompt_state = gr.State(system_prompt)
        system_prompt_input.change(lambda prompt: prompt, inputs=[system_prompt_input], outputs=[system_prompt_state])

    return deepsearch_tab


def create_tavily_tab(model, tokenizer, assistant_marker, system_prompt, temperature):
    """Creates the UI components and logic for the Tavily Search tab."""
    logger.info("Creating Tavily Search Tab")
    # tokenizer_for_template = cast(PreTrainedTokenizer, tokenizer) # Global now

    # --- Tavily Client Setup ---
    tavily_api_key = os.getenv("TAVILY_API_KEY")
    if not tavily_api_key:
        logger.error("TAVILY_API_KEY not found in environment variables.")
        with gr.Blocks(analytics_enabled=False) as tavily_tab_error:
            gr.Markdown("# ⚠️ Tavily Search Error")
            gr.Markdown("TAVILY_API_KEY environment variable not set. Please set it and restart the application.")
        return tavily_tab_error

    try:
        tavily_client = TavilyClient(api_key=tavily_api_key)
        logger.info("TavilyClient initialized successfully.")
    except Exception as e:
        logger.error(f"Failed to initialize TavilyClient: {e}")
        with gr.Blocks(analytics_enabled=False) as tavily_tab_error:
            gr.Markdown("# ⚠️ Tavily Client Initialization Error")
            gr.Markdown(f"Failed to initialize Tavily Client: {e}")
        return tavily_tab_error

    # --- Agent Streaming Logic for Tavily ---
    def stream_tavily_agent_response(
        message: str,
        history_gr: list[gr.ChatMessage],
        temp: float,
        max_iter: int = 20,
        num_search_results: int = 2,  # Tavily default/recommendation might differ
    ) -> Iterator[list[gr.ChatMessage]]:
        """Stream agent responses using Tavily for search."""
        local_tokenizer_for_template = _tokenizer_for_template_global  # Use global
        assert local_tokenizer_for_template is not None

        chat_state = {
            "messages": [{"role": "system", "content": system_prompt}],
            "finished": False,
        }
        processed_history = history_gr[:-1] if history_gr else []
        for msg_obj in processed_history:
            role = getattr(msg_obj, "role", "unknown")
            content = getattr(msg_obj, "content", "")
            if role == "user":
                chat_state["messages"].append({"role": "user", "content": build_user_prompt(content)})
            elif role == "assistant":
                chat_state["messages"].append({"role": "assistant", "content": content})

        chat_state["messages"].append({"role": "user", "content": build_user_prompt(message)})

        initial_token_length = get_chat_num_tokens(chat_state, local_tokenizer_for_template)
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

            current_length_before_gen = get_chat_num_tokens(chat_state, local_tokenizer_for_template)
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
                formatted_prompt = apply_chat_template(chat_state, tokenizer=local_tokenizer_for_template)["text"]
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
                        content=f"Searching (Tavily) for: {search_query}",
                        metadata={"title": "🔍 ReZero Query", "status": "pending"},
                    )
                )
                yield messages
                search_start = time.time()
                try:
                    # --- Tavily Search Call ---
                    logger.info(f"Performing Tavily search for: {search_query}")
                    tavily_response = tavily_client.search(
                        query=search_query,
                        search_depth="advanced",
                        max_results=num_search_results,
                        include_answer=False,
                        include_raw_content=False,
                    )
                    search_duration = time.time() - search_start
                    logger.info(f"Tavily search completed in {search_duration:.2f}s.")

                    # --- Format Tavily Results ---
                    results_list = tavily_response.get("results", [])
                    formatted_tavily_results = ""
                    if results_list:
                        formatted_tavily_results = "\n".join(
                            [
                                f"Doc {i + 1} (Title: {res.get('title', 'N/A')}) URL: {res.get('url', 'N/A')}\n{res.get('content', '')}"
                                for i, res in enumerate(results_list)
                            ]
                        )
                    else:
                        formatted_tavily_results = "No results found by Tavily."

                    messages[search_msg_idx] = gr.ChatMessage(
                        role="assistant",
                        content=f"{search_query}",
                        metadata={"title": "🔍 ReZero Query", "duration": search_duration},
                    )
                    yield messages

                    display_results = formatted_tavily_results
                    messages.append(
                        gr.ChatMessage(
                            role="assistant",
                            content=display_results,
                            metadata={"title": "ℹ️ Tavily Information", "status": "done"},
                        )
                    )
                    yield messages

                    formatted_results_for_llm = f"<information>{formatted_tavily_results}</information>"
                    chat_state["messages"].append({"role": "user", "content": formatted_results_for_llm})

                except Exception as e:
                    search_duration = time.time() - search_start
                    logger.error(f"Tavily Search failed: {str(e)}")
                    messages[search_msg_idx] = gr.ChatMessage(
                        role="assistant",
                        content=f"Tavily Search failed: {str(e)}",
                        metadata={"title": "❌ Tavily Search Error", "status": "done", "duration": search_duration},
                    )
                    yield messages
                    chat_state["messages"].append(
                        {"role": "system", "content": f"Error during Tavily search: {str(e)}"}
                    )
                    chat_state["finished"] = True

            current_length_after_iter = get_chat_num_tokens(chat_state, local_tokenizer_for_template)
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

    # --- UI Layout for Tavily Tab ---
    with gr.Blocks(analytics_enabled=False) as tavily_tab:
        gr.Markdown("# 🌐 Tavily Search with Visible Thinking")
        gr.Markdown("Ask questions answered using the Tavily web search API.")

        with gr.Row():
            with gr.Column(scale=3):
                tavily_chatbot = gr.Chatbot(
                    [],
                    elem_id="tavily_chatbot",
                    type="messages",
                    height=600,
                    show_label=False,
                    render_markdown=True,
                    bubble_full_width=False,
                )
                tavily_msg = gr.Textbox(
                    placeholder="Type your message here...",
                    show_label=False,
                    container=False,
                    elem_id="tavily_msg-input",
                )
                tavily_example_questions = [
                    "What is the weather like in London today?",
                    "Summarize the latest news about AI advancements.",
                    "Who won the last Formula 1 race?",
                ]
                gr.Examples(
                    examples=tavily_example_questions,
                    inputs=tavily_msg,
                    label="Example Questions (Web Search)",
                    examples_per_page=3,
                )
                with gr.Row():
                    tavily_clear = gr.Button("Clear Chat")
                    tavily_submit = gr.Button("Submit", variant="primary")

            with gr.Column(scale=1):
                gr.Markdown("### Settings")
                tavily_temp_slider = gr.Slider(
                    minimum=0.1, maximum=1.0, value=temperature, step=0.1, label="Temperature"
                )
                tavily_system_prompt_input = gr.Textbox(
                    label="System Prompt", value=system_prompt, lines=3, info="Controls how the AI behaves"
                )
                tavily_max_iter_slider = gr.Slider(
                    minimum=1,
                    maximum=20,
                    value=10,
                    step=1,
                    label="Max Search Iterations",
                    info="Maximum number of search-think cycles",
                )
                tavily_num_results_slider = gr.Slider(
                    minimum=1,
                    maximum=5,
                    value=3,
                    step=1,
                    label="Number of Search Results",
                    info="How many results to retrieve per search query",
                )

        # --- Event Handlers for Tavily Tab ---
        def tavily_add_user_message(
            user_msg_text: str, history: list[gr.ChatMessage]
        ) -> tuple[str, list[gr.ChatMessage]]:
            if user_msg_text and user_msg_text.strip():
                history.append(gr.ChatMessage(role="user", content=user_msg_text.strip()))
            return "", history

        tavily_submitted_msg_state = gr.State("")

        tavily_submit.click(
            lambda msg_text: msg_text,
            inputs=[tavily_msg],
            outputs=[tavily_submitted_msg_state],
            queue=False,
        ).then(
            tavily_add_user_message,
            inputs=[tavily_submitted_msg_state, tavily_chatbot],
            outputs=[tavily_msg, tavily_chatbot],
            queue=False,
        ).then(
            stream_tavily_agent_response,  # Use Tavily-specific stream function
            inputs=[
                tavily_submitted_msg_state,
                tavily_chatbot,
                tavily_temp_slider,
                tavily_max_iter_slider,
                tavily_num_results_slider,
            ],
            outputs=tavily_chatbot,
        )

        tavily_msg.submit(
            lambda msg_text: msg_text,
            inputs=[tavily_msg],
            outputs=[tavily_submitted_msg_state],
            queue=False,
        ).then(
            tavily_add_user_message,
            inputs=[tavily_submitted_msg_state, tavily_chatbot],
            outputs=[tavily_msg, tavily_chatbot],
            queue=False,
        ).then(
            stream_tavily_agent_response,  # Use Tavily-specific stream function
            inputs=[
                tavily_submitted_msg_state,
                tavily_chatbot,
                tavily_temp_slider,
                tavily_max_iter_slider,
                tavily_num_results_slider,
            ],
            outputs=tavily_chatbot,
        )

        tavily_clear.click(lambda: ([], ""), None, [tavily_chatbot, tavily_submitted_msg_state])

        tavily_system_prompt_state = gr.State(system_prompt)
        tavily_system_prompt_input.change(
            lambda prompt: prompt, inputs=[tavily_system_prompt_input], outputs=[tavily_system_prompt_state]
        )

    return tavily_tab


def main():
    """Run the Gradio app with tabs."""
    model_path = str(GENERATOR_MODEL_DIR)
    logger.info(f"Using model from config: {model_path}")

    # Shared model setup (do once)
    try:
        model, tokenizer, assistant_marker = setup_model_and_tokenizer(model_path)
    except Exception as e:
        logger.critical(f"Failed to load model/tokenizer: {e}")
        # Display error if model fails to load
        with gr.Blocks() as demo:
            gr.Markdown("# Critical Error")
            gr.Markdown(
                f"Failed to load model or tokenizer from '{model_path}'. Check the path and ensure the model exists.\n\nError: {e}"
            )
            demo.launch(share=True)
        sys.exit(1)  # Exit if model loading fails

    system_prompt = get_system_prompt()
    default_temp = 0.7

    # Define tokenizer_for_template globally after successful load
    global _tokenizer_for_template_global
    _tokenizer_for_template_global = cast(PreTrainedTokenizer, tokenizer)

    # Create content for each tab
    tab1 = create_deepsearch_tab(model, tokenizer, assistant_marker, system_prompt, default_temp)
    tab2 = create_tavily_tab(model, tokenizer, assistant_marker, system_prompt, default_temp)

    # Combine tabs
    interface = gr.TabbedInterface(
        [tab1, tab2], tab_names=["ReZero (VectorDB)", "Tavily Search (Web)"], title="ReZero Demo"
    )

    logger.info("Launching Gradio Tabbed Interface...")
    interface.launch(share=True)


if __name__ == "__main__":
    if load_vectorstore() is None:
        logger.warning("⚠️ FAISS vectorstore could not be loaded. Search functionality may be unavailable.")

    main()
