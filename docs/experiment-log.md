# Experiment Log

## 250404-llama-3.2-3b-instruct-grpo-03

- Experiment assets: <https://huggingface.co/janhq/250404-llama-3.2-3b-instruct-grpo-03>
- Base model: lllama-3.2-3b-instruct
- Max agent turns: 10
- reward_functions:
    - reward_correctness
    - reward_format
    - reward_em_chunk
    - (NEW) reward_retry (with better logic): from just reward for number of search attemps, to reward for number of search attempts but ONLY when there are an answer (bascially no matter how hard the llm tried, no result = 0 reward 💀)
    - reward_search_strategy
    - reward_search_quality
- reward weight: [4.0, 2.0, 1.0, 1.0, 1.0, 1.0] (dont ask me why the weight is like this, my ancestor told me so)
- (NEW) Max agent turns: 20: We hypothesized that the model is not given enough turns to give the answer, so we increase the max agent turns from 10 to 20

- Evaluation results on 32 samples
    - Base:  15.62%
    - 50: 21.88%
    - 100: 28.12%
    - 150: 28.12%
    - 200: 37.50%
    - **250: 46.88%**
    - 300: 31.25%
    - 350: 12.50%
    - 400: 18.75%
    - 450: 0.00%
    - 500: 3.12%
    - 550: 0.00%
    - 600:
    - 650:
- Observation:
    - The model achived much better result than the previous experiment, at step 250.
    - The loss isn't crashed this time, but the reward still crashed after step 350.

## 250404-llama-3.2-3b-instruct-grpo-02

- Experiment assets: <https://huggingface.co/janhq/250404-llama-3.2-3b-instruct-grpo-02>
- Base model: lllama-3.2-3b-instruct
- Max agent turns: 10
- reward_functions:
    - reward_correctness
    - reward_format
    - reward_em_chunk
    - reward_retry
    - (NEW) reward_search_strategy: reward if the reasoning steps use words that tailored for searching
    - (NEW) reward_search_quality: reward if the search queries in the same conversation are diverse (low reward if they are similar to each other)
- (NEW) reward weight: [4.0, 2.0, 1.0, 1.0, 1.0, 1.0] (dont ask me why the weight is like this, my ancestor told me so)
- (NEW) Max agent turns: 20

- Evaluation results on 32 samples
    - Base: 15.62%
    - 100: 18.75%
    - 200: 18.75%
    - 300: 21.88%
**- 400: 31.25%**
    - 500: 18.75%
    - 600: 0
    - 700: 0
    - 800: 0
    - 900: 0
    - 1000: 0
- Observation:

## 250404-llama-3.2-3b-instruct-grpo-01

- Experiment assets: <https://huggingface.co/janhq/250404-llama-3.2-3b-instruct-grpo-01>
- Base model: lllama-3.2-3b-instruct
- Max agent turns: 10
- reward_functions:
    - reward_correctness
    - reward_format
    - reward_em_chunk
    - reward_retry
- reward_weights: all equal 1.0
- This experiment is train and evaluated with bugged `reward_correctness` function so the result is not reliable (the reward_correctness got non final answer as input (<information> or <search> for example), and still compare that input with)
- Observation: even though the reward_correctness is bugged, the model reward still goes up, but the final result is not good

## Design of reward function

- `reward_format`: check for correct json tool parsing
- `reward_correctness`: use the llm itself to verify the generated answer against the ground truth
- `reward_em_chunk`: check for exact match of the retrieved chunk against the ground truth chunk that is used to make the ground truth answer
- `reward_retry`: reward base on number of search calls to encourage more searches, capped the reward at about 5 searches

## Redesign of prompt template

### Original prompts

```python
def get_system_prompt():
    """Get the system prompt with current date."""
    current_date = datetime.now().strftime("%d %b %Y")
    return f"""Cutting Knowledge Date: December 2023
Today Date: {current_date}

When you receive a tool call response, use the output to format an answer to the original user question.

You are a helpful assistant with tool calling capabilities.
"""


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
                    "description": "The query to search the knowledge corpus with"
                },
            },
            "required": ["query"]
        }
    }
}

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

You may also reason in any message, thinking step by step about how to answer the question. Wrap your reasoning in <reasoning> and </reasoning> tags.

{json.dumps(SEARCH_TOOL_DEFINITION, indent=2)}

Question: {q}
"""
    return user_prompt
```

### Edit 1 (move from json tool call to simple <search> and </search> tags)

```python
def get_system_prompt():
    """Get the system prompt with current date."""
    current_date = datetime.now().strftime("%d %b %Y")
    return f"""Cutting Knowledge Date: December 2023
Today Date: {current_date}

You are a helpful assistant with search capabilities.
"""


user_prompt = f"""Answer the given question. \
You must conduct reasoning inside <think> and </think> first every time you get new information. \
After reasoning, if you find you lack some knowledge, you can call a search engine by <search> query </search>. \
Based on the user's core intent, formulate the most effective search query using specific, descriptive keywords that differentiate the topic clearly. \
Aim for queries that resemble how an expert searcher might phrase it, like using "compare lithium-ion vs solid-state battery efficiency" rather than just "batteries". \
The document will be provided inside <information> and </information> tags to you later. \
You can search as many turns as you want, but only one search query per turn. \
If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer>, without detailed illustrations. \
Only answer when you have 100% confidence in the search results, else continue searching. \
Question: {q}\n"""
```

### Edit 2 (Better)

This edit explicitly tells the model to follow the desired format and do not introduce <information> </information> tags. The result is that the reward format increase much faster and more stable. Also, the model does not hallucinate the <information> </information> tags.

This edit is made parallel with the edit logic of reward_format with stricter checking.

```python
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
```

## Initial Experiments

- Starting from running the exact training script from Autodiact
- Some observations:
    - Only 2 reward functions:
        - `reward_format`: check for correct json tool parsing
        - `reward_correctness`: use the llm itself to verify the generated answer against the ground truth
        - Training for 101 steps, the reward did go up and the accuracy improved
- New: Start adding 2 more reward functions:
    - `reward_em_chunk`: check for exact match of the retrieved chunk against the ground truth chunk that is used to make the ground truth answer
    - `reward_retry`: reward base on number of search calls to encourage more searches, capped the reward at about 5 searches
- Observations after adding the 2 new reward functions: The reward and accuracy still go up as normal, didn't observe any thing special

### Dataset

- The full data is generated from the <https://github.com/menloresearch/DeepSearch/blob/main/scripts/generate_data.py> script
    - The training datasets consist of  309 samples
    - The validation datasets consist of 32 samples
    - This sample dataset is used for all experiments up til now.
  