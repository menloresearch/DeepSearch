# Chat Template 101

This repo was orignally created with the chat template of LLama instruct model family, so i need to somehow hackaround to be able to train new models base on deepseek-r1-distil-xxx

## Getting the intuition

- <https://huggingface.co/docs/transformers/main/chat_templating>
- > A chat template is **a part of the tokenizer** and it specifies how to convert conversations into a single tokenizable string in the expected model format.

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1")
chat = [
  {"role": "user", "content": "Hello, how are you?"},
  {"role": "assistant", "content": "I'm doing great. How can I help you today?"},
  {"role": "user", "content": "I'd like to show off how chat templating works!"},
]

tokenizer.apply_chat_template(chat, tokenize=False)

<s>[INST] Hello, how are you? [/INST]I'm doing great. How can I help you today?</s> [INST] I'd like to show off how chat templating works! [/INST]
```

- 💡 OHhhhh can just make a jupyter notebook to play around with this
