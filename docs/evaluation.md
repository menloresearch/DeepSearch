# Evaluation

- **Goal**:
    - 1. Better performance than the original one (by auto eval script)
    - 2. Better performance by real human eval/preference

## Implementation Phases

- [x] 1. Just take the eval function from the original repo (it simply uses accuracy (exact match)) and simple quick glance on the output quality.
- [ ] 2. Find some more common and conventional dataset and benchmarks (still auto script)
- [ ] 3. Setup human eval

## Baseline

- Info from autodidact
    - After just 100 steps of GRPO training (1 hour on a single RTX 4090 GPU), Llama-8B significantly improved its ability to research and answer questions from the Apollo 13 mission report
    - On a validation set of 68 questions, accuracy more than doubled from 23% to 59%.

- Training log: idk why but the result that I got from acutally running the training is a bit lower.

```bash
ceTensor.clone().detach() or sourceTensor.clone().detach().requires_grad_(True), rather than torch.tensor(sourceTensor).
  completion_ids = [torch.tensor(ids, device=device) for ids in completion_ids]
Processed prompts: 100%|████████████████| 16/16 [00:00<00:00, 39.27it/s, est. speed input: 6827.13 toks/s, output: 81.01 toks/s]
rewards_per_func: tensor([0.6875, 0.7000], device='cuda:0'):05,  2.55it/s, est. speed input: 385.80 toks/s, output: 5.11 toks/s]
{'loss': 0.0003, 'grad_norm': 0.5810762047767639, 'learning_rate': 0.0, 'rewards/reward_correctness': 0.6875, 'rewards/reward_formatting': 0.699999988079071, 'reward': 1.3875000476837158, 'reward_std': 0.44403791427612305, 'completion_length': 224.125, 'kl': 0.00834659393876791, 'epoch': 0.34}
{'train_runtime': 7992.2854, 'train_samples_per_second': 0.202, 'train_steps_per_second': 0.013, 'train_loss': 0.0005197484556535774, 'epoch': 0.34}
100%|███████████████████████████████████████████████████████████████████████████████████████| 101/101 [2:13:12<00:00, 79.13s/it]
Processed prompts: 100%|████████████████| 67/67 [00:20<00:00,  3.28it/s, est. speed input: 950.44 toks/s, output: 394.51 toks/s]
Processed prompts: 100%|███████████████| 66/66 [00:20<00:00,  3.15it/s, est. speed input: 2383.55 toks/s, output: 323.82 toks/s]
Processed prompts: 100%|███████████████| 20/20 [00:17<00:00,  1.13it/s, est. speed input: 1320.49 toks/s, output: 146.76 toks/s]
Processed prompts: 100%|████████████████| 17/17 [00:16<00:00,  1.04it/s, est. speed input: 1620.28 toks/s, output: 98.35 toks/s]
Processed prompts: 100%|██████████████████| 9/9 [00:15<00:00,  1.73s/it, est. speed input: 1165.77 toks/s, output: 71.38 toks/s]
Processed prompts: 100%|████████████████| 67/67 [00:04<00:00, 16.31it/s, est. speed input: 3617.28 toks/s, output: 61.11 toks/s]
RESULTS:
percentage of correct answers: 0.5074626865671642
==============================
Processed prompts: 100%|███████████████| 67/67 [00:15<00:00,  4.46it/s, est. speed input: 1292.29 toks/s, output: 561.32 toks/s]
Processed prompts: 100%|███████████████| 44/44 [00:18<00:00,  2.44it/s, est. speed input: 1800.84 toks/s, output: 244.13 toks/s]
Processed prompts: 100%|███████████████| 13/13 [00:12<00:00,  1.05it/s, est. speed input: 1209.04 toks/s, output: 126.32 toks/s]
Processed prompts: 100%|███████████████| 10/10 [00:13<00:00,  1.32s/it, est. speed input: 1225.46 toks/s, output: 109.78 toks/s]
Processed prompts: 100%|██████████████████| 7/7 [00:12<00:00,  1.86s/it, est. speed input: 1149.18 toks/s, output: 76.05 toks/s]
Processed prompts: 100%|████████████████| 67/67 [00:02<00:00, 31.53it/s, est. speed input: 6047.70 toks/s, output: 83.31 toks/s]
RESULTS:
percentage of correct answers: 0.19402985074626866
==============================
[rank0]:[W320 07:13:50.651270455 ProcessGroupNCCL.cpp:1250] Warning: WARNING: process group has NOT been destroyed before we destruct ProcessGroupNCCL. On normal program exit, the application should call destroy_process_group to ensure that any pending NCCL operations have finished in this process. In rare cases this process can exit before this point and block the progress of another member of the process group. This constraint has always been present,  but this warning has only been added since PyTorch 2.4 (function operator())
```

- Training log with paraphrased dataset (no new reward function yet!) - Disappointing results

  ```bash

.2587745785713196, 'completion_length': 374.3125, 'kl': 0.004571444820612669, 'epoch': 0.34}
{'train_runtime': 7419.1437, 'train_samples_per_second': 0.218, 'train_steps_per_second': 0.014, 'train_loss': 0.00037626780881639505, 'epoch': 0.34}
100%|████████████████████████████████████████████████████████| 101/101 [2:03:39<00:00, 73.46s/it]
Processed prompts: 100%|█| 67/67 [00:19<00:00,  3.51it/s, est. speed input: 1016.34 toks/s, outpu
Processed prompts: 100%|█| 66/66 [00:21<00:00,  3.03it/s, est. speed input: 2086.78 toks/s, outpu
Processed prompts: 100%|█| 19/19 [00:14<00:00,  1.28it/s, est. speed input: 1326.10 toks/s, outpu
Processed prompts: 100%|█| 14/14 [00:14<00:00,  1.03s/it, est. speed input: 1363.04 toks/s, outpu
Processed prompts: 100%|█| 9/9 [00:13<00:00,  1.55s/it, est. speed input: 1153.10 toks/s, output:
Processed prompts: 100%|█| 67/67 [00:02<00:00, 28.46it/s, est. speed input: 5843.91 toks/s, outpu
RESULTS:
percentage of correct answers: 0.3582089552238806
==============================

Processed prompts: 100%|█| 67/67 [00:20<00:00,  3.20it/s, est. speed input: 925.56 toks/s, output
Processed prompts: 100%|█| 36/36 [00:13<00:00,  2.63it/s, est. speed input: 1755.08 toks/s, outpu
Processed prompts: 100%|█| 11/11 [00:09<00:00,  1.19it/s, est. speed input: 1254.10 toks/s, outpu
Processed prompts: 100%|█| 8/8 [00:09<00:00,  1.15s/it, est. speed input: 1192.77 toks/s, output:
Processed prompts: 100%|█| 4/4 [00:06<00:00,  1.67s/it, est. speed input: 1063.38 toks/s, output:
Processed prompts: 100%|█| 67/67 [00:02<00:00, 29.78it/s, est. speed input: 5244.11 toks/s, outpu
RESULTS:
percentage of correct answers: 0.2835820895522388
==============================

[rank0]:[W324 11:21:27.955684565 ProcessGroupNCCL.cpp:1250] Warning: WARNING: process group has NOT been destroyed before we destruct ProcessGroupNCCL. On normal program exit, the application should call destroy_process_group to ensure that any pending NCCL operations have finished in this process. In rare cases this process can exit before this point and block the progress of another member of the process group. This constraint has always been present,  but this warning has only been added since PyTorch 2.4 (function operator())

  ```

## Getting some sense of the eval data or benchmark

- > For example, benchmarks like ARC-AGI, which involve visual reasoning, remain challenging for these models, even though they might seem straightforward to a human. (ichigo)
