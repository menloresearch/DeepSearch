"""Test agent functionality."""

from transformers import LlamaTokenizerFast

from src.deepsearch.agent import Agent
from src.deepsearch.tokenizer_adapter import LlamaTokenizerAdapter, R1DistilTokenizerAdapter


def mock_generate_fn(prompts):
    """Mock generation function that returns simple responses."""

    class MockResponse:
        def __init__(self, text):
            self.outputs = [type("obj", (object,), {"text": text})()]

    return [MockResponse(f"Assistant: Test response for {i}") for i, _ in enumerate(prompts)]


def test_llama_agent_response_mask_lengths():
    """Test that response tokens and masks have the same length for Llama."""
    # Test data
    questions = ["What is Python?", "How to write tests?"]

    # Setup Llama agent
    tokenizer = LlamaTokenizerFast.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")
    agent = Agent(LlamaTokenizerAdapter())

    # Run agent
    outputs = agent.run_agent(
        generate_fn=mock_generate_fn, tokenizer=tokenizer, questions=questions, max_generations=1, max_new_tokens=100
    )

    # Check lengths match for each example
    for i, (tokens, mask) in enumerate(zip(outputs.response_tokens, outputs.response_masks)):
        print(f"\nExample {i}:")
        print(f"Question: {questions[i]}")
        print(f"Response tokens length: {len(tokens)}")
        print(f"Response mask length: {len(mask)}")

        assert len(tokens) == len(mask), f"Mismatch in example {i}: tokens={len(tokens)}, mask={len(mask)}"
        assert mask.sum().item() > 0, "Mask should have some 1s indicating response tokens"
        assert all(x in [0, 1] for x in mask.tolist()), "Mask should only contain 0s and 1s"


def test_r1_distil_agent_response_mask_lengths():
    """Test that response tokens and masks have the same length for R1-Distil."""
    # Test data
    questions = ["What is Python?", "How to write tests?"]

    # Setup R1-Distil agent
    tokenizer = LlamaTokenizerFast.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
    agent = Agent(R1DistilTokenizerAdapter())

    # Run agent
    outputs = agent.run_agent(
        generate_fn=mock_generate_fn, tokenizer=tokenizer, questions=questions, max_generations=1, max_new_tokens=100
    )

    # Check lengths match for each example
    for i, (tokens, mask) in enumerate(zip(outputs.response_tokens, outputs.response_masks)):
        print(f"\nExample {i}:")
        print(f"Question: {questions[i]}")
        print(f"Response tokens length: {len(tokens)}")
        print(f"Response mask length: {len(mask)}")

        assert len(tokens) == len(mask), f"Mismatch in example {i}: tokens={len(tokens)}, mask={len(mask)}"
        assert mask.sum().item() > 0, "Mask should have some 1s indicating response tokens"
        assert all(x in [0, 1] for x in mask.tolist()), "Mask should only contain 0s and 1s"
