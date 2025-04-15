<div align="center">

# ReZero: Enhancing LLM search ability by trying one-more-time

<img width="2128" alt="ReZeroer" src="https://github.com/user-attachments/assets/f1145fe2-a452-4f23-ab0e-f1518c16aa55" />

ReZero trains a small language model to develop effective search behaviors instead of memorizing static data. It interacts with multiple synthetic search engines, each with unique retrieval mechanisms, to refine queries and persist in searching until it finds exact answers. The project focuses on reinforcement learning, preventing overfitting, and optimizing for efficiency in real-world search applications.

[**Quick Demo**](#quick-demo-) | [**Setup**](#setup-️) | [**Data and Training**](#data-and-training-) | [**Models**](#models-) | [**References**](#references-) | [**Acknowledgements**](#acknowledgements-)

</div>

## Quick Demo 🚀

<div align="center">
  <img src="assets/demo.gif" alt="ReZero Demo" width="1000px">
</div>

Run the interactive web interface to see ReZero in action:

```bash
python app.py
```

This will launch a Gradio interface where you can interact with the model and test different search behaviors.

## Setup 🛠️

```bash
# Clone the repository
git clone https://github.com/menloresearch/ReZero
cd ReZero

# Create virtual environment
python -m venv .venv

# Activate the environment
source .venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -e .

# Set up environment variables (required for websearch demo)
cp .env.example .env
# Edit .env and add your Tavily API key if you want to use the websearch demo
```

## Data and Training 🧠

All necessary training data is included in the `data/` folder. To train:

```bash
python train_grpo.py
```

If you want to regenerate the data, please run:

```bash
python scripts/generate_data.py
```

## Models 🤖

You can find our models on Hugging Face 🤗! We're committed to open-source and easy access for the research community.

| Model | Backbone | Size | Link |
|-------|----------|------|------|
| ReZero-v0.1 | Llama-3.2-3B | 3B | [🤗 Menlo/ReZero-v0.1-llama-3.2-3b-it-grpo-250404](https://huggingface.co/Menlo/ReZero-v0.1-llama-3.2-3b-it-grpo-250404) |

## References 📖

## Acknowledgements 🤝

- This project is kickstarted from the source code of [AutoDidact](https://github.com/dCaples/AutoDidact)
