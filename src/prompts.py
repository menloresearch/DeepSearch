"""
Prompt-related functions for handling system and user prompts.
"""

from datetime import datetime


def get_system_prompt():
    """Get the system prompt with current date."""
    current_date = datetime.now().strftime("%d %b %Y")
    return f"""Cutting Knowledge Date: December 2023
Today Date: {current_date}

You are a helpful assistant with search capabilities.
"""


def build_user_prompt(q):
    """
    Build a user prompt with the question using the new template format.

    Args:
        q (str): The question to ask

    Returns:
        str: Formatted user prompt
    """
    user_prompt = f"""Answer the given question. \
You must conduct reasoning inside <think> and </think> first every time you get new information. \
You ONLY HAVE TWO CHOICES after thinking: to search or to answer but not both.
If you find you lack some knowledge, you MUST call a search engine by <search> query </search>. 
Based on the user's core intent, formulate the most effective search query using specific, descriptive keywords that differentiate the topic clearly. \
Aim for queries that resemble how an expert searcher might phrase it, like using "compare lithium-ion vs solid-state battery efficiency" rather than just "batteries". \
You can search as many turns as you want, but only one search query per thinking. \
The information will be provided when you end your response. \
If you find no further external knowledge needed, you MUST directly provide the answer inside <answer> and </answer>, without detailed illustrations. \
You can only answer one time, so make sure to answer when you have 100% confidence in the search results, else continue searching. \
You MUST END YOUR RESPONSE WITH either <answer> and </answer> tags or <search> and </search> tags. \
Question: {q}\n"""
    return user_prompt


def format_search_results(results: str | list[str]) -> str:
    """
    Format search results for display, matching the format from infer.py.
    Each result should be in the format: "Doc X(Title: Y) content"

    Args:
        results: Search results as string or list of strings

    Returns:
        Formatted search results with document titles
    """
    if isinstance(results, list):
        # If results are already in the correct format, just join them
        if any("Doc" in r and "Title:" in r for r in results):
            content = "\n".join(results)
        else:
            # If results are raw content, format them with default titles
            content = "\n".join([f"Doc {i + 1}(Title: Document {i + 1})\n{r}" for i, r in enumerate(results)])
    else:
        # If single result is already formatted, use it as is
        if "Doc" in results and "Title:" in results:
            content = results
        else:
            # If single result is raw content, format it with default title
            content = f"Doc 1(Title: Document 1)\n{results}"

    return content
