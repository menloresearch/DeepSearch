# Worklog

## Backlog

- [ ] Modify `generate_dataset.py` (**ONLY AFTER** the simple training and benchmark works):
    - [ ] As a data dataset maker, I want to change from LLama 3.1 8B to API call, like claude, gemini or openai. Originally they use 3.1 8B for `Self-Bootstrapping` demonstration, but the dataset quality is low, for sure.
    - [ ] Experimenting with different chunking strategies
- [ ] [search-backends.md](search-backends.md) design (for more dataset noise (**ONLY AFTER** the simple training dataset works))

- [ ] Research a little bit on Agentic Reward Modeling (for designing better reward function maybe?)
    - <https://medium.com/@techsachin/agentic-reward-modeling-combine-human-preferences-with-verifiable-correctness-signals-for-reliable-76c408b3491c>
    - <https://arxiv.org/pdf/2502.19328>
    - <https://github.com/THU-KEG/Agentic-Reward-Modeling>
    - <https://www.themoonlight.io/en/review/agentic-reward-modeling-integrating-human-preferences-with-verifiable-correctness-signals-for-reliable-reward-systems>

## yymmdd

- [ ] task description

## 250324

- [ ] @thinhlpg transfers the project to @bachvudinh

## 250323

- [ ] Train the model
- [ ] Make the dataset
- [ ] Upload datasets to HF Hub
      - Initial dataset from AutoDidact
      - Paraphrased sdataset
- [ ] Make a simple gradio demo app

## 250322

- [x] Moving all the scattered and disorganized stuffs that've been working on for the past week into this repo.
- [x] Write  proposal for DeepSearch
    - [x] [evaluation.md](evaluation.md) design (list out the metrics and why)
    - [x] [dataset.md](dataset.md) design (pipeline, data structure,...)
    - [x] [reward-functions.md](reward-functions.md) design (list out the functions and why)
- [x] As a new member of research team, i'm curious on how did we do GRPO with Alphamaze?, so that I can inherit the good stuff and improve the workflow!!!
    - [Alphamaze](https://github.com/menloresearch/visual-thinker)?
    - <https://www.menlo.ai/blog/alpha-maze>
    - <https://arxiv.org/pdf/2502.14669>
    - > Our training process involved two key stages: creating a specialized dataset and then using a combination of supervised fine-tuning (SFT) and reinforcement learning (RL) to train the model.
    - LLaMA-Factory for SFT **(1.5B 6xA6000 1.5 hour)** and Unsloth for GRPO
    - 💡 Hmm so for SFT we have 50% successful data and 50% retry data, and full successful data for GRPO. Can I also apply this to DeepSearch as well? #HACK

## 250321

- [x] Inspect the code of AutoDidact in a more detailed way <https://github.com/menloresearch/DeepSearch/issues/4>

## 250320

- Research on GRPO <https://github.com/menloresearch/DeepSearch/issues/2>

## 250319

- Research on GRPO <https://github.com/menloresearch/DeepSearch/issues/2>
- Run the training script of AutoDidact

## 250318

- Idea received <https://github.com/menloresearch/DeepSearch/issues/1>
