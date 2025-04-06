"""
Test cases for reward functions in rewards.py
"""

import pytest

from src.deepsearch.rewards import (
    build_reward_correctness_fn,
    reward_em_chunk,
    reward_format,
    reward_retry,
    reward_search_diversity,
    reward_search_strategy,
)


class MockResponse:
    """Mock response class that simulates vLLM response"""

    def __init__(self, text):
        self.outputs = [type("obj", (object,), {"text": text})()]


# Mock functions for testing
def mock_vllm_generate_func(*args, **kwargs):
    """Mock function that returns verification responses based on the input"""
    # Check if the prompt contains "5" (wrong answer) or "4" (correct answer)
    prompt = str(args[0]) if args else ""
    if "5" in prompt:
        return [MockResponse("No, the answer is incorrect")]  # Return False for wrong answer
    return [MockResponse("Yes, the answer is correct")]  # Return True for correct answer


class MockTokenizer:
    """Mock tokenizer class that simulates the behavior of a real tokenizer"""

    def __init__(self):
        self.input_ids = [1, 2, 3]

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        """Mock apply_chat_template method"""
        # For testing, we just return a formatted string
        return "\n".join([f"{msg['role']}: {msg['content']}" for msg in messages])


@pytest.fixture
def reward_correctness_fn():
    """Fixture to create reward correctness function"""
    return build_reward_correctness_fn(mock_vllm_generate_func, MockTokenizer())


def test_reward_correctness_basic(reward_correctness_fn):
    """Test basic reward correctness functionality"""
    prompts = ["What is 2+2?"]
    completions = [{"messages": [{"role": "assistant", "content": "<answer>4</answer>"}]}]
    reward_kwargs = {"answer": ["4"]}

    rewards = reward_correctness_fn(prompts, completions, **reward_kwargs)
    assert len(rewards) == 1  # Should return one verification result per answer
    assert rewards[0] is True  # Should be True for correct answer


def test_reward_correctness_wrong_answer(reward_correctness_fn):
    """Test reward correctness with wrong answer"""
    prompts = ["What is 2+2?"]
    completions = [{"messages": [{"role": "assistant", "content": "<answer>5</answer>"}]}]
    reward_kwargs = {"answer": ["4"]}

    rewards = reward_correctness_fn(prompts, completions, **reward_kwargs)
    assert len(rewards) == 1  # Should return one verification result per answer
    assert rewards[0] is False  # Should be False for wrong answer


def test_reward_format_correct():
    """Test reward format with correct format"""
    prompts = ["Test prompt"]
    completions = [
        {
            "messages": [
                {"role": "assistant", "content": "<think>\nSome reasoning\n</think>\n<answer>\nThe answer\n</answer>"}
            ]
        }
    ]
    rewards = reward_format(prompts, completions)
    assert rewards[0] == 1.0


def test_reward_format_with_search():
    """Test reward format with search tags only (no answer tags)"""
    prompts = ["Test prompt"]
    completions = [
        {"messages": [{"role": "assistant", "content": "<think>\nSome reasoning\n</think>\n<search>query</search>"}]}
    ]
    rewards = reward_format(prompts, completions)
    assert rewards[0] == 1.0


def test_reward_format_markdown_tags():
    """Test reward format with markdown-styled tags"""
    prompts = ["Test prompt"]
    markdown_formats = [
        {
            "messages": [
                {
                    "role": "assistant",
                    "content": "**<think>**\nSome reasoning\n**</think>**\n<answer>\nThe answer\n</answer>",
                }
            ]
        },
        {
            "messages": [
                {
                    "role": "assistant",
                    "content": "*<think>*\nSome reasoning\n*</think>*\n<answer>\nThe answer\n</answer>",
                }
            ]
        },
        {
            "messages": [
                {
                    "role": "assistant",
                    "content": "_<think>_\nSome reasoning\n_</think>_\n<answer>\nThe answer\n</answer>",
                }
            ]
        },
    ]

    for completion in markdown_formats:
        rewards = reward_format(["Test prompt"], [completion])
        assert rewards[0] == 0.0, f"Failed with: {completion['messages'][0]['content']}"


def test_reward_format_information_tags():
    """Test reward format with information tags"""
    prompts = ["Test prompt"]
    # Test different information tag variants
    info_variants = [
        "<information>Some info</information>",
        "<info>Some info</info>",
        "<Info>Some info</Info>",
        "<INFORMATION>Some info</INFORMATION>",
        "<INFO>Some info</INFO>",
    ]

    for info_tag in info_variants:
        content = f"<think>\nSome reasoning\n</think>\n{info_tag}\n<answer>\nThe answer\n</answer>"
        completions = [{"messages": [{"role": "assistant", "content": content}]}]
        rewards = reward_format(prompts, completions)
        assert rewards[0] == 0.0, f"Failed to detect information tag: {info_tag}"


def test_reward_format_real_example():
    """Test reward format with a real-world example - should fail now since it has both search and answer tags"""
    prompts = ["What cars did Paul Walker drive in Fast and Furious?"]
    content = """<think>I need to search for Paul Walker's cars in Fast and Furious movies.</think>
<search> Paul Walker's cars in Fast and Furious </search>

From the information provided, it's clear that Paul Walker was a part of the "Fast and Furious" series, but the specific list of cars is not mentioned. Since I lack this particular detail, I will call a search engine to get the specific list of cars Paul Walker drove in the "Fast and Furious" movies.

<search> list of cars paul walker drove in Fast and Furious </search>

Based on the updated information, it seems the focus was on his career, financials, and family. However, I am still missing the specific list of cars he drove in the "Fast and Furious" movies. Since it appears that the information might not be contained within the accessed documents, and I have no further search queries to make, I will provide an answer based on the details I have.

<answer> Charger </answer>"""

    completions = [{"messages": [{"role": "assistant", "content": content}]}]
    rewards = reward_format(prompts, completions)
    assert rewards[0] == 0.0, "Should reject responses with both search and answer tags"


def test_reward_format_real_example_search_only():
    """Test reward format with search-only format in a real-world example"""
    prompts = ["What cars did Paul Walker drive in Fast and Furious?"]
    content = """<think>I need to search for Paul Walker's cars in Fast and Furious movies.</think>
<search> Paul Walker's cars in Fast and Furious </search>"""

    completions = [{"messages": [{"role": "assistant", "content": content}]}]
    rewards = reward_format(prompts, completions)
    assert rewards[0] == 1.0, "Should accept responses with only search tags"


def test_reward_format_real_example_answer_only():
    """Test reward format with answer-only format in a real-world example"""
    prompts = ["What cars did Paul Walker drive in Fast and Furious?"]
    content = """<think>Based on the information provided, it seems Paul Walker drove a Charger in the Fast and Furious series.</think>
<answer> Charger </answer>"""

    completions = [{"messages": [{"role": "assistant", "content": content}]}]
    rewards = reward_format(prompts, completions)
    assert rewards[0] == 1.0, "Should accept responses with only answer tags"


def test_reward_format_incorrect_tag_sequence():
    """Test reward format with incorrect tag sequence - should fail since we require proper sequence and ending"""
    formats = [
        {
            "messages": [
                {"role": "assistant", "content": "<answer>\nThe answer\n</answer>\n<think>\nSome reasoning\n</think>"}
            ]
        },
        {
            "messages": [
                {
                    "role": "assistant",
                    "content": "<search>query</search>\n<think>\nSome reasoning\n</think>",
                }
            ]
        },
    ]

    for completion in formats:
        rewards = reward_format([], [completion])
        assert rewards[0] == 0.0, f"Should fail with incorrect sequence: {completion['messages'][0]['content']}"

    # Test correct sequences
    correct_formats = [
        {
            "messages": [
                {"role": "assistant", "content": "<think>\nSome reasoning\n</think>\n<answer>\nThe answer\n</answer>"}
            ]
        },
        {
            "messages": [
                {
                    "role": "assistant",
                    "content": "<think>\nSome reasoning\n</think>\n<search>query</search>",
                }
            ]
        },
    ]

    for completion in correct_formats:
        rewards = reward_format([], [completion])
        assert rewards[0] == 1.0, f"Should pass with correct sequence: {completion['messages'][0]['content']}"


def test_reward_format_multiple_answers():
    """Test reward format with multiple answer tags"""
    completions = [
        {
            "messages": [
                {
                    "role": "assistant",
                    "content": "<think>\nSome reasoning\n</think>\n<answer>\nFirst answer\n</answer>\n<answer>\nSecond answer\n</answer>",
                }
            ]
        }
    ]
    rewards = reward_format([], completions)
    assert rewards[0] == 0.0


def test_reward_format_incomplete_tags():
    """Test reward format with incomplete tags"""
    incomplete_formats = [
        {
            "messages": [
                {"role": "assistant", "content": "<think>\nMissing closing think tag\n<answer>\nThe answer\n</answer>"}
            ]
        },
        {
            "messages": [
                {
                    "role": "assistant",
                    "content": "<think>\nSome reasoning\n</think>\n<answer>\nMissing closing answer tag",
                }
            ]
        },
        {
            "messages": [
                {
                    "role": "assistant",
                    "content": "Missing opening think tag\n</think>\n<answer>\nThe answer\n</answer>",
                }
            ]
        },
    ]

    for completion in incomplete_formats:
        rewards = reward_format([], [completion])
        assert rewards[0] == 0.0, f"Failed with: {completion['messages'][0]['content']}"


def test_reward_retry():
    """Test reward retry function"""
    prompts = ["What is the capital of France?"]
    completions = [
        {
            "messages": [
                {"role": "assistant", "content": "<think>Let me search</think>\n<search>capital of France</search>"},
                {"role": "assistant", "content": "<think>Need more info</think>\n<search>Paris history</search>"},
                {"role": "assistant", "content": "<think>Found it</think>\n<answer>Paris</answer>"},
            ]
        }
    ]
    rewards = reward_retry(prompts, completions)
    assert len(rewards) == 1
    assert rewards[0] > 0, "Should give positive reward for multiple search attempts"


def test_reward_em_chunk():
    """Test exact match chunk reward function"""
    prompts = ["What is Python?"]
    completions = [
        {"messages": [{"role": "user", "content": "<information>Python is a programming language</information>"}]}
    ]
    correct_contents = ["Python is a programming language"]

    rewards = reward_em_chunk(prompts, completions, chunk_content=correct_contents)
    assert len(rewards) == 1
    assert rewards[0] == 1.0, "Should give full reward for exact chunk match"


def test_reward_em_chunk_no_chunk_content():
    """Test reward EM chunk with no chunk content provided"""
    completions = [{"messages": [{"role": "ipython", "content": "<information>Some content</information>"}]}]

    with pytest.raises(ValueError, match="chunk_content must be provided"):
        reward_em_chunk([], completions)


def test_reward_em_chunk_multiple_chunks():
    """Test reward EM chunk with multiple chunks to match"""
    completions = [
        {"messages": [{"role": "ipython", "content": "<information>First chunk content</information>"}]},
        {"messages": [{"role": "user", "content": "<information>Second chunk content</information>"}]},
    ]
    reward_kwargs = {"chunk_content": ["First chunk content", "Second chunk content"]}

    rewards = reward_em_chunk([], completions, **reward_kwargs)
    assert len(rewards) == 2
    assert rewards == [1.0, 1.0], "Should get reward 1.0 for each matching chunk"


def test_reward_em_chunk_whitespace_handling():
    """Test reward EM chunk handles whitespace properly"""
    completions = [
        {"messages": [{"role": "ipython", "content": "  <information>  Content with spaces  </information>  "}]}
    ]
    reward_kwargs = {"chunk_content": ["Content with spaces"]}

    rewards = reward_em_chunk([], completions, **reward_kwargs)
    assert rewards[0] == 1.0, "Should handle whitespace in content and tags"


def test_reward_format_search_or_answer_not_both():
    """Test that having both search and answer tags in the same message is not allowed"""
    content = "<think>I need to search</think>\n<search>query</search>\n<answer>Final answer</answer>"
    completions = [{"messages": [{"role": "assistant", "content": content}]}]
    rewards = reward_format([], completions)
    assert rewards[0] == 0.0, "Should reject messages with both search and answer tags"

    # Verify that having just search tag is valid
    content_search_only = "<think>I need to search</think>\n<search>query</search>"
    completions = [{"messages": [{"role": "assistant", "content": content_search_only}]}]
    rewards = reward_format([], completions)
    assert rewards[0] == 1.0, "Should accept messages with just search tags"

    # Verify that having just answer tag is valid
    content_answer_only = "<think>I know the answer</think>\n<answer>Final answer</answer>"
    completions = [{"messages": [{"role": "assistant", "content": content_answer_only}]}]
    rewards = reward_format([], completions)
    assert rewards[0] == 1.0, "Should accept messages with just answer tags"


def test_reward_correctness_validation(reward_correctness_fn):
    """Test reward correctness validation logic for message roles and tags"""
    prompts = ["What is 2+2?"]
    test_cases = [
        # Test assistant role validation
        {
            "completion": {"messages": [{"role": "user", "content": "<answer>4</answer>"}]},
            "expected": False,
            "desc": "Non-assistant role should fail",
        },
        # Test answer tag validation
        {
            "completion": {"messages": [{"role": "assistant", "content": "4"}]},
            "expected": False,
            "desc": "Missing answer tags should fail",
        },
        # Test search tag validation
        {
            "completion": {"messages": [{"role": "assistant", "content": "<answer>4</answer><search>query</search>"}]},
            "expected": False,
            "desc": "Having search tags should fail",
        },
        # Test information tag validation
        {
            "completion": {
                "messages": [{"role": "assistant", "content": "<answer>4</answer><information>info</information>"}]
            },
            "expected": False,
            "desc": "Having information tags should fail",
        },
        # Test valid case
        {
            "completion": {"messages": [{"role": "assistant", "content": "<answer>4</answer>"}]},
            "expected": True,
            "desc": "Valid format should pass",
        },
    ]

    for case in test_cases:
        rewards = reward_correctness_fn(prompts, [case["completion"]], answer=["4"])
        assert rewards[0] == case["expected"], f"Failed: {case['desc']}"


def test_reward_search_strategy_perfect():
    """Test search strategy reward with a perfect search strategy and info processing."""
    content = [
        {
            "role": "assistant",
            "content": """<think>Let me search for a broad overview first.</think>
<search>what is quantum computing overview</search>""",
        },
        {
            "role": "user",
            "content": "<information>Quantum computing uses quantum mechanics principles...</information>",
        },
        {
            "role": "assistant",
            "content": """<think>Based on the provided information about quantum mechanics, I should look for specific examples.</think>
<search>quantum computing practical examples</search>""",
        },
        {
            "role": "user",
            "content": "<information>Some practical examples of quantum computing include...</information>",
        },
        {
            "role": "assistant",
            "content": """<think>According to the examples provided, I should investigate the advantages.</think>
<search>quantum computing advantages over classical computing</search>""",
        },
        {"role": "user", "content": "<information>The key advantages of quantum computing are...</information>"},
        {
            "role": "assistant",
            "content": """<think>Based on all the information gathered about quantum computing principles, examples, and advantages, I can now provide a comprehensive answer.</think>
<answer>Quantum computing is a revolutionary technology that...</answer>""",
        },
    ]

    completions = [{"messages": content}]
    rewards = reward_search_strategy(prompts=["test"], completions=completions)
    assert rewards[0] == 1.0  # Initial (0.2) + info processing (0.4) + final synthesis (0.4)


def test_reward_search_strategy_no_refinement():
    """Test search strategy reward with just initial search."""
    content = [
        {
            "role": "assistant",
            "content": """<think>Let me search for information.</think>
<search>quantum computing</search>""",
        }
    ]

    completions = [{"messages": content}]
    rewards = reward_search_strategy(prompts=["test"], completions=completions)
    assert rewards[0] == 0.2  # Only initial search reward


def test_reward_search_strategy_multiple_refinements():
    """Test search strategy reward with multiple info-based refinements."""
    content = [
        {
            "role": "assistant",
            "content": """<think>Starting with a broad search.</think>
<search>what is machine learning</search>""",
        },
        {"role": "user", "content": "<information>Machine learning is a branch of AI...</information>"},
        {
            "role": "assistant",
            "content": """<think>Based on this overview, let's look for specific applications.</think>
<search>machine learning examples in healthcare</search>""",
        },
        {"role": "user", "content": "<information>In healthcare, ML is used for diagnosis...</information>"},
        {
            "role": "assistant",
            "content": """<think>According to the healthcare examples, we should explore different methods.</think>
<search>supervised vs unsupervised learning in medical diagnosis</search>""",
        },
    ]

    completions = [{"messages": content}]
    rewards = reward_search_strategy(prompts=["test"], completions=completions)
    assert rewards[0] == pytest.approx(0.6)  # Initial (0.2) + info processing (0.4)


def test_reward_search_strategy_search_operators():
    """Test search strategy reward with search operators and info processing."""
    content = [
        {
            "role": "assistant",
            "content": """<think>Let's use specific search operators.</think>
<search>"exact phrase" site:edu</search>""",
        },
        {"role": "user", "content": "<information>Educational resources show that...</information>"},
        {
            "role": "assistant",
            "content": """<think>Based on the educational resources, we need more specific results.</think>
<search>machine learning -basic filetype:pdf</search>""",
        },
        {"role": "user", "content": "<information>Advanced ML concepts include...</information>"},
        {
            "role": "assistant",
            "content": """<think>According to these findings, let's combine key concepts.</think>
<search>AI AND "deep learning" OR "neural networks"</search>""",
        },
    ]

    completions = [{"messages": content}]
    rewards = reward_search_strategy(prompts=["test"], completions=completions)
    assert rewards[0] == pytest.approx(0.6)  # Initial (0.2) + info processing (0.4)


def test_reward_search_strategy_non_assistant_messages():
    """Test search strategy reward with mixed message roles."""
    content = [
        {"role": "user", "content": "<search>user search</search>"},
        {
            "role": "assistant",
            "content": """<think>Let me search for information.</think>
<search>what is quantum computing</search>""",
        },
        {"role": "system", "content": "<think>system thinking</think>"},
    ]

    completions = [{"messages": content}]
    rewards = reward_search_strategy(prompts=["test"], completions=completions)
    assert rewards[0] == 0.2  # Only counts the assistant's initial search


def test_reward_search_strategy_final_synthesis():
    """Test search strategy reward with final synthesis but no refinements."""
    content = [
        {
            "role": "assistant",
            "content": """<think>Let me search for information.</think>
<search>quantum computing basics</search>""",
        },
        {"role": "user", "content": "<information>Quantum computing uses qubits...</information>"},
        {
            "role": "assistant",
            "content": """<think>Based on the information about quantum computing basics, I can now provide an answer.</think>
<answer>Quantum computing is a field that...</answer>""",
        },
    ]

    completions = [{"messages": content}]
    rewards = reward_search_strategy(prompts=["test"], completions=completions)
    assert rewards[0] == pytest.approx(0.6)  # Initial (0.2) + final synthesis (0.4)


def test_reward_search_diversity_no_search():
    """Test search diversity reward with no search queries."""
    content = [
        {
            "role": "assistant",
            "content": "<think>Let me think about this.</think>",
        }
    ]
    completions = [{"messages": content}]
    rewards = reward_search_diversity(prompts=["test"], completions=completions)
    assert rewards[0] == 0.0


def test_reward_search_diversity_single_query():
    """Test search diversity reward with a single search query."""
    content = [
        {
            "role": "assistant",
            "content": """<think>Let me search.</think>
<search>what is python programming</search>""",
        }
    ]
    completions = [{"messages": content}]
    rewards = reward_search_diversity(prompts=["test"], completions=completions)
    assert rewards[0] == pytest.approx(0.2)  # Base reward for single query


def test_reward_search_diversity_diverse_queries():
    """Test search diversity reward with diverse search queries."""
    content = [
        {
            "role": "assistant",
            "content": """<think>Let's start broad.</think>
<search>what is machine learning</search>""",
        },
        {
            "role": "assistant",
            "content": """<think>Now specific applications.</think>
<search>healthcare diagnosis using neural networks</search>""",
        },
        {
            "role": "assistant",
            "content": """<think>Let's look at a different aspect.</think>
<search>ethical concerns in AI decision making</search>""",
        },
    ]
    completions = [{"messages": content}]
    rewards = reward_search_diversity(prompts=["test"], completions=completions)
    # Should get high reward due to diverse queries
    assert rewards[0] > 0.5


def test_reward_search_diversity_exact_duplicates():
    """Test search diversity reward with exact duplicate queries."""
    content = [
        {
            "role": "assistant",
            "content": """<think>First search.</think>
<search>python tutorial</search>""",
        },
        {
            "role": "assistant",
            "content": """<think>Searching again.</think>
<search>python tutorial</search>""",
        },
        {
            "role": "assistant",
            "content": """<think>One more time.</think>
<search>python tutorial</search>""",
        },
    ]
    completions = [{"messages": content}]
    rewards = reward_search_diversity(prompts=["test"], completions=completions)
    # Should get very low reward due to exact duplicates
    assert rewards[0] < 0.2


def test_reward_retry_no_answer():
    """Test reward_retry when final message has no answer tags - should return 0."""
    prompts = ["Test prompt"]
    completions = [
        {
            "messages": [
                {"role": "assistant", "content": "<think>Let me search</think><search>query 1</search>"},
                {"role": "assistant", "content": "<think>Let me search again</think><search>query 2</search>"},
            ]
        }
    ]
    rewards = reward_retry(prompts, completions)
    assert rewards[0] == 0.0, "Should return 0 when final message has no answer tags"


def test_reward_retry_with_answer():
    """Test reward_retry with answer in final message - should calculate reward normally."""
    prompts = ["Test prompt"]
    completions = [
        {
            "messages": [
                {"role": "assistant", "content": "<think>Let me search</think><search>query 1</search>"},
                {"role": "assistant", "content": "<think>Let me search again</think><search>query 2</search>"},
                {"role": "assistant", "content": "<think>Here's the answer</think><answer>Final answer</answer>"},
            ]
        }
    ]
    rewards = reward_retry(prompts, completions)
    expected = round(0.2 + 2 * 0.15, 3)  # base_reward + 2 searches * increment
    assert rewards[0] == expected, "Should calculate reward normally when final message has answer tags"


def test_reward_retry_violation_with_answer():
    """Test reward_retry with multiple searches in one message but answer in final message."""
    prompts = ["Test prompt"]
    completions = [
        {
            "messages": [
                {
                    "role": "assistant",
                    "content": "<think>Multiple searches</think><search>query 1</search><search>query 2</search>",
                },
                {"role": "assistant", "content": "<think>Here's the answer</think><answer>Final answer</answer>"},
            ]
        }
    ]
    rewards = reward_retry(prompts, completions)
    expected = round((0.2 + 2 * 0.15) * (1 - 0.5), 3)  # (base + 2*increment) * (1 - violation_penalty)
    assert rewards[0] == expected, "Should apply violation penalty but still calculate reward due to answer"


def test_reward_retry_optimal_searches():
    """Test reward_retry with optimal number of searches and answer."""
    prompts = ["Test prompt"]
    completions = [
        {
            "messages": [
                {"role": "assistant", "content": "<think>Search 1</think><search>query 1</search>"},
                {"role": "assistant", "content": "<think>Search 2</think><search>query 2</search>"},
                {"role": "assistant", "content": "<think>Search 3</think><search>query 3</search>"},
                {"role": "assistant", "content": "<think>Search 4</think><search>query 4</search>"},
                {"role": "assistant", "content": "<think>Search 5</think><search>query 5</search>"},
                {"role": "assistant", "content": "<think>Here's the answer</think><answer>Final answer</answer>"},
            ]
        }
    ]
    rewards = reward_retry(prompts, completions)
    expected = round(0.2 + 5 * 0.15, 3)  # base_reward + optimal_search_count * increment
    assert rewards[0] == expected, "Should cap reward at optimal search count"


def test_reward_retry_beyond_optimal():
    """Test reward_retry with more than optimal searches but still with answer."""
    prompts = ["Test prompt"]
    completions = [
        {
            "messages": [
                {"role": "assistant", "content": "<think>Search 1</think><search>query 1</search>"},
                {"role": "assistant", "content": "<think>Search 2</think><search>query 2</search>"},
                {"role": "assistant", "content": "<think>Search 3</think><search>query 3</search>"},
                {"role": "assistant", "content": "<think>Search 4</think><search>query 4</search>"},
                {"role": "assistant", "content": "<think>Search 5</think><search>query 5</search>"},
                {"role": "assistant", "content": "<think>Search 6</think><search>query 6</search>"},
                {"role": "assistant", "content": "<think>Here's the answer</think><answer>Final answer</answer>"},
            ]
        }
    ]
    rewards = reward_retry(prompts, completions)
    expected = round(0.2 + 5 * 0.15, 3)  # base_reward + optimal_search_count * increment
    assert rewards[0] == expected, "Should not exceed max reward even with more searches"


def test_reward_retry_empty_messages():
    """Test reward_retry with empty message list."""
    prompts = ["Test prompt"]
    completions = [{"messages": []}]
    rewards = reward_retry(prompts, completions)
    assert rewards[0] == 0.0, "Should return 0 for empty message list"


def test_reward_retry_no_searches_with_answer():
    """Test reward_retry with no searches but has answer."""
    prompts = ["Test prompt"]
    completions = [
        {
            "messages": [
                {"role": "assistant", "content": "<think>Direct answer</think><answer>Final answer</answer>"},
            ]
        }
    ]
    rewards = reward_retry(prompts, completions)
    assert rewards[0] == 0.0, "Should return 0 when no searches even with answer"


def test_reward_retry_multiple_completions():
    """Test reward_retry with multiple completions."""
    prompts = ["Test 1", "Test 2"]
    completions = [
        {
            "messages": [
                {"role": "assistant", "content": "<think>Search</think><search>query</search>"},
                {"role": "assistant", "content": "<think>Answer</think><answer>Final 1</answer>"},
            ]
        },
        {
            "messages": [
                {"role": "assistant", "content": "<think>Search 1</think><search>query 1</search>"},
                {"role": "assistant", "content": "<think>Search 2</think><search>query 2</search>"},
                {"role": "assistant", "content": "<think>Answer</think><answer>Final 2</answer>"},
            ]
        },
    ]
    rewards = reward_retry(prompts, completions)
    expected1 = round(0.2 + 0.15, 3)  # base_reward + 1 search * increment
    expected2 = round(0.2 + 2 * 0.15, 3)  # base_reward + 2 searches * increment
    assert rewards == [expected1, expected2], "Should handle multiple completions correctly"
