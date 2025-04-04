# DeepSearch - A Hard Working Search Engine 🔍
<img width="2128" alt="DeepSearcher" src="https://github.com/user-attachments/assets/f1145fe2-a452-4f23-ab0e-f1518c16aa55" />

DeepSearch trains a small language model to develop effective search behaviors instead of memorizing static data. It interacts with multiple synthetic search engines, each with unique retrieval mechanisms, to refine queries and persist in searching until it finds exact answers. The project focuses on reinforcement learning, preventing overfitting, and optimizing for efficiency in real-world search applications.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Evaluation

Compare base model with LoRA-enhanced model performance:

```bash
# Quick run with defaults
./eval.sh

# Custom run
./eval.sh --lora_path "/path/to/lora" --temperature 0.7
```

Direct Python usage:

```bash
python eval.py --lora_path "/path/to/lora" --temperature 0.7
```

The tool generates a results file with accuracy metrics and improvement statistics.

## Models

You can find our models on Hugging Face 🤗! We're committed to open-source and easy access for the research community.

| Model | Backbone | Size | Link |
|-------|----------|------|------|
| - | - | - | - |

## Datasets

We've released our datasets on Hugging Face 🤗 to support reproducibility and further research.

| Dataset                             | Description                                         | Size  | Link                                                                                    |
|--------------------------------------|-----------------------------------------------------|-------|-----------------------------------------------------------------------------------------|
| -                                    | -                                                   | -     | -                                                                                       |
| -                                    | -                                                   | -     | -                                                                                       |
| -                                    | -                                                   | -     | -                                                                                       |

## References

- This project is kickstarted from [AutoDidact](https://github.com/dCaples/AutoDidact)

## Personal Notes

- **This is research code**, so I'm prioritizing speed over code quality for now. Expect things to be messy—both the code and commit history. Roasting is welcome, but don't judge me too hard; I'll clean it up later. **I don't know what I don't know**, but I'm eager (and desperate) to learn and improve, so any constructive feedback is highly appreciated! 💖
