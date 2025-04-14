# DeepSearch - A Hard Working Search Engine 🔍

<img width="2128" alt="DeepSearcher" src="https://github.com/user-attachments/assets/f1145fe2-a452-4f23-ab0e-f1518c16aa55" />

DeepSearch trains a small language model to develop effective search behaviors instead of memorizing static data. It interacts with multiple synthetic search engines, each with unique retrieval mechanisms, to refine queries and persist in searching until it finds exact answers. The project focuses on reinforcement learning, preventing overfitting, and optimizing for efficiency in real-world search applications.

## Quick Demo 🚀

Run the interactive web interface to see DeepSearch in action:

```bash
python app.py
```

This will launch a Gradio interface where you can interact with the model and test different search behaviors.

You can also evaluate model performance:

```bash
# Using the evaluation scripts
python scripts/eval_lora.py --lora_path "/path/to/lora"
python scripts/eval_base.py
```

## Setup 🛠️

1. Clone the repository with submodules:

```bash
git clone --recurse-submodules [repository-url]
cd DeepSearch
```

2. Set up your environment variables:

```bash
cp .env.example .env
# Edit .env to add your HuggingFace token and OpenRouter API key
```

3. Install dependencies using the development setup:

```bash
make install
```

This installs the project in editable mode along with all dependencies specified in pyproject.toml, including:

- transformers
- unsloth
- gradio
- langchain
- and other required packages

## Data Preparation 📊

DeepSearch uses the Musique dataset for training and evaluation.

### Download and prepare all data in one step

```bash
make prepare-all-musique
```

### Step-by-step data preparation

1. Download the Musique dataset:

   ```bash
   make download-musique
   ```

2. Prepare the JSONL files for training:

   ```bash
   make prepare-musique-jsonl
   ```

3. Extract paragraphs for indexing:

   ```bash
   make extract-musique-paragraphs
   ```

4. Build the FAISS index:

   ```bash
   make build-musique-index
   ```

5. Prepare development data:

   ```bash
   make prepare-dev-data
   ```

6. Validate data preparation:

   ```bash
   make check-data
   ```

## Training 🧠

Train the model using the GRPO (General Reinforcement Learning from Outer Preferences) approach:

```bash
python train_grpo.py
```

You can monitor training progress with TensorBoard:

```bash
make tensorboard
```

List available training runs:

```bash
make list-runs
```

## Development 💻

### Run tests

```bash
make test
```

### Code quality and style

```bash
# Format code
make style

# Check code quality
make quality

# Auto-fix issues
make fix
```

### Clean up

```bash
make clean
```

## Models 🤖

You can find our models on Hugging Face 🤗! We're committed to open-source and easy access for the research community.

| Model | Backbone | Size | Link |
|-------|----------|------|------|
| - | - | - | - |

## Datasets 📚

We've released our datasets on Hugging Face 🤗 to support reproducibility and further research.

| Dataset | Description | Size | Link |
|---------|-------------|------|------|
| - | - | - | - |
| - | - | - | - |
| - | - | - | - |

## References 📖

- This project is kickstarted from [AutoDidact](https://github.com/dCaples/AutoDidact)
