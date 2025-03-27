# Worklog

## Backlog

- [ ] @thinhlpg transfers the project to @bachvudinh
- [ ] Modify `generate_dataset.py` (**ONLY AFTER** the simple training and benchmark works):
    - [ ] Optimize speed (different LLM models, api, tools, etc.)
    - [ ] Optimize quality. As a data dataset maker, I want to change from LLama 3.1 8B to API call, like claude, gemini or openai. Originally they use 3.1 8B for `Self-Bootstrapping` demonstration, but the dataset quality is low, for sure.
    - [ ] Experimenting with different chunking strategies
- [ ] [search-backends.md](search-backends.md) design (for more dataset noise (**ONLY AFTER** the simple training dataset works))

- [ ] Train SFT first stage, then GRPO (new idea from @tikikun 250326)
    - I think this idea is already implemented in search-r1 repo, i'll double check it later.
- [ ]  Implement quality of life scripts from [brain-rotting-multiple-gpu-workflow-for-dummies.md](brain-rotting-multiple-gpu-workflow-for-dummies.md)
- [ ] Better verification logic please (should be a fixed for every experiments, not the base model it self)

## yymmdd

- [ ] task description

## 250329

- brain.exe and back.exe refused to work

## 250328

- [ ] Watch solo leveling with bro  @tikikun 🔥
- [ ] Figuring out how to keep multiple experiments organized. the repos in the server are a mess 💀💀 (but at least they worked for now)

## 250328 - ❗❗❗D-Day❗❗❗

- [ ] Show the results, or demo

## 250327

- [x] CLEAN THE REPO PLEASE IT'S A MESS 😭😭😭
    - Double checked all script, runned well :3
- [ ] Write script to train x-deepseek-r1-distil models (original script only support Llama -instruct models)
- [ ] Script to continue training from last checkpoint
- [ ] Make a simple demo app (or just cli inference script should be good)
- [ ] Upload datasets to HF Hub
- [ ] Research a little bit on Agentic Reward Modeling (for designing better reward function maybe?) [agentic-reward-modeling.md](agentic-reward-modeling.md)

## 250326

- Fix exact match reward function bug
- Enhance the training script with better logging and monitoring
- Train new models
- Write new eval script

## 250325

- [x] Read Search-R1 to get more ideas on how to improve the reward functions (pretty similar idea i suppose)
- [x] update new reward functions in [reward-functions.md](reward-functions.md)
- [x] Train the model v0 (with new data and reward functions) (might be another 2 hours)
    - spoiler: it's not good

## 250324

- [x] Make the dataset v0
- [x] Train with new data and default reward functions (it took 2 hours on 1xA6000 😭)
    - Got poor result (50% Accuracy down to 35%) 📉

## 250323

- brain.exe and back.exe refused to work 😭

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

## Graveyard 💀

- ~~Convert this notebook to script [250324_generate_data_anatomy.ipynb](../notebooks/250324_generate_data_anatomy.ipynb)~~ (no need, already have a script for that)
