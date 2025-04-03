# Evaluation

- **Goal**:
    - 1. Better performance than the original one (by auto eval script)
    - 2. Better performance by real human eval/preference

## Benmarks

Just go with this 4 for now:

- HotpotQA
- 2wiki
- Musique
- Bamboogle

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

- LLama3 1B on my local machine, with new retry reward function

```

torch.tensor(ids, device=device) for ids in completion_ids
Processed prompts: 100%|█████| 8/8 [00:03<00:00,  2.33it/s, est. speed input: 611.91 toks/s, output: 285.83 toks/s]
rewards_per_func: tensor([0.1250, 0.1750, 0.0225], device='cuda:0')eed input: 611.91 toks/s, output: 285.83 toks/s]
{'loss': 0.0001, 'grad_norm': 0.5529439449310303, 'learning_rate': 0.0, 'rewards/reward_correctness': 0.125, 'rewards/reward_formatting': 0.17499999701976776, 'rewards/reward_retry_behavior': 0.02252296172082424, 'reward': 0.32252296805381775, 'reward_std': 0.6055484414100647, 'completion_length': 333.125, 'kl': 0.002497631125152111, 'epoch': 0.17}
{'train_runtime': 2145.4442, 'train_samples_per_second': 0.377, 'train_steps_per_second': 0.047, 'train_loss': 7.476110755337125e-05, 'epoch': 0.17}
100%|████████████████████████████████████████████████████████████████████████████| 101/101 [35:45<00:00, 21.24s/it]
Processed prompts: 100%|██████████████████| 67/67 [01:27<00:00,  1.30s/it, est. speed input: 221.09 toks/s, output: 446.71 toks/s]
Processed prompts:  20%|▏| 2/10 [00:02<00:11,  1.45s/it, est. speed input: 713.29 toks/s, output: 4Processed prompts: 100%|███████████████| 10/10 [00:06<00:00,  1.51it/s, est. speed input: 1464.06 toks/s, output: 255.67 toks/s]
Processed prompts: 100%|██████████████████| 1/1 [00:00<00:00,  2.20it/s, est. speed input: 3494.66 toks/s, output: 59.45 toks/s]
Processed prompts: 100%|███████████████| 67/67 [00:03<00:00, 18.46it/s, est. speed input: 5495.01 toks/s, output: 154.33 toks/s]
RESULTS:
percentage of correct answers: 0.3283582089552239
==============================

Processed prompts: 100%|████████████████| 67/67 [01:20<00:00,  1.21s/it, est. speed input: 238.22 toks/s, output: 529.58 toks/s]
Processed prompts: 100%|███████████████| 14/14 [00:06<00:00,  2.20it/s, est. speed input: 2025.55 toks/s, output: 315.92 toks/s]
Processed prompts: 100%|█████████████████| 3/3 [00:00<00:00,  3.05it/s, est. speed input: 4166.43 toks/s, output: 125.21 toks/s]
Processed prompts: 100%|███████████████| 67/67 [00:04<00:00, 16.35it/s, est. speed input: 6068.44 toks/s, output: 184.25 toks/s]
RESULTS:
percentage of correct answers: 0.29850746268656714
==============================

[rank0]:[W325 18:27:54.262290956 ProcessGroupNCCL.cpp:1496] Warning: WARNING: destroy_process_group() was not called before program exit, which can leak resources. For more info, please see <https://pytorch.org/docs/stable/distributed.html#shutdown> (function operator())
(.venv) (base)

```

- LLama3 1B with new data and 4 reward functions

```bash
2025-03-25 21:59:47.993 | DEBUG    | rl_helpers:check_student_answers:493 - Verification result: True
2025-03-25 21:59:47.993 | DEBUG    | rl_helpers:check_student_answers:493 - Verification result: True
2025-03-25 21:59:47.993 | DEBUG    | rl_helpers:check_student_answers:493 - Verification result: False
2025-03-25 21:59:47.993 | DEBUG    | rl_helpers:check_student_answers:493 - Verification result: False
2025-03-25 21:59:47.993 | INFO     | rl_helpers:check_student_answers:495 - Verification complete. 11/32 answers correct
2025-03-25 21:59:47.994 | INFO     | rl_helpers:run_eval:634 - EVALUATION RESULTS:
2025-03-25 21:59:47.994 | INFO     | rl_helpers:run_eval:635 - Percentage of correct answers: 0.344
2025-03-25 21:59:47.994 | INFO     | rl_helpers:run_eval:636 - ==============================
[rank0]:[W325 21:59:48.406952498 ProcessGroupNCCL.cpp:1496] Warning: WARNING: destroy_process_group() was not called before program exit, which can leak resources. For more info, please see https://pytorch.org/docs/stable/distributed.html#shutdown (function operator())
(.venv) (base) 
~/code/DeepSearch  dev ✗    
```

- Llama3  7B with new data and 4 reward functions (bro wtf :())

```bash
2025-03-25 17:07:05.533 | DEBUG    | rl_helpers:check_student_answers:493 - Verification result: False
2025-03-25 17:07:05.533 | DEBUG    | rl_helpers:check_student_answers:493 - Verification result: False
2025-03-25 17:07:05.533 | DEBUG    | rl_helpers:check_student_answers:493 - Verification result: False
2025-03-25 17:07:05.533 | DEBUG    | rl_helpers:check_student_answers:493 - Verification result: False
2025-03-25 17:07:05.533 | INFO     | rl_helpers:check_student_answers:495 - Verification complete. 1/32 answers correct
2025-03-25 17:07:05.535 | INFO     | rl_helpers:run_eval:634 - EVALUATION RESULTS:
2025-03-25 17:07:05.535 | INFO     | rl_helpers:run_eval:635 - Percentage of correct answers: 0.031
2025-03-25 17:07:05.535 | INFO     | rl_helpers:run_eval:636 - ==============================
[rank0]:[W325 17:07:06.452081140 ProcessGroupNCCL.cpp:1250] Warning: WARNING: process group has NOT been destroyed before we destruct ProcessGroupNCCL. On normal program exit, the application should call destroy_process_group to ensure that any pending NCCL operations have finished in this process. In rare cases this process can exit before this point and block the progress of another member of the process group. This constraint has always been present,  but this warning has only been added since PyTorch 2.4 (function operator())
```

- Llama3 7B with new data only

```bash
2025-03-25 16:48:33.168 | DEBUG    | rl_helpers:check_student_answers:493 - Verification result: False
2025-03-25 16:48:33.168 | DEBUG    | rl_helpers:check_student_answers:493 - Verification result: True
2025-03-25 16:48:33.168 | DEBUG    | rl_helpers:check_student_answers:493 - Verification result: True
2025-03-25 16:48:33.168 | DEBUG    | rl_helpers:check_student_answers:493 - Verification result: False
2025-03-25 16:48:33.168 | DEBUG    | rl_helpers:check_student_answers:493 - Verification result: False
2025-03-25 16:48:33.168 | DEBUG    | rl_helpers:check_student_answers:493 - Verification result: True
2025-03-25 16:48:33.168 | INFO     | rl_helpers:check_student_answers:495 - Verification complete. 9/32 answers correct
2025-03-25 16:48:33.176 | INFO     | rl_helpers:run_eval:634 - EVALUATION RESULTS:
2025-03-25 16:48:33.177 | INFO     | rl_helpers:run_eval:635 - Percentage of correct answers: 0.281
2025-03-25 16:48:33.177 | INFO     | rl_helpers:run_eval:636 - ==============================
[rank0]:[W325 16:48:34.303740078 ProcessGroupNCCL.cpp:1250] Warning: WARNING: process group has NOT been destroyed before we destruct ProcessGroupNCCL. On normal program exit, the application should call destroy_process_group to ensure that any pending NCCL operations have finished in this process. In rare cases this process can exit before this point and block the progress of another member of the process group. This constraint has always been present,  but this warning has only been added since PyTorch 2.4 (function operator())
```

```bash
=== Evaluation 2025-03-26 14:06:01 ===
Model: meta-llama/Llama-3.2-1B-Instruct
Checkpoint: trainer_output_meta-llama_Llama-3.2-1B-Instruct_gpu0_20250326_140600/checkpoint-101
Trained model accuracy: 0.281 (28.1%)
Base model accuracy: 0.188 (18.8%)
I mprovement: 0.094 (9.4%)

```

```bash
=== Evaluation 2025-03-26 15:25:13 ===
Model: meta-llama/Llama-3.1-8B-Instruct
Checkpoint: trainer_output_meta-llama_Llama-3.1-8B-Instruct_gpu1_20250326_134236/checkpoint-101
Trained model accuracy: 0.281 (28.1%)
Base model accuracy: 0.188 (18.8%)
Improvement: 0.094 (9.4%)

```

- waht the f*ck they have the same accuracy??? this is really bullshit.

- new eval script

```bash
orrect
Sample outputs saved to trainer_output_meta-llama_Llama-3.1-8B-Instruct_gpu0_20250326_223903/lora_model_debug_outputs.txt

Evaluation of LoRA model completed
Accuracy: 0.7500
Results saved to ./test_results/20250326_223637/model_comparison_results.txt

Model comparison completed.
Base Model Accuracy: 0.6562
LoRA Model Accuracy: 0.7500
Improvement: 0.0938
Results saved to ./test_results/20250326_223637/model_comparison_results.txt
[rank0]:[W326 22:42:43.889848919 ProcessGroupNCCL.cpp:1250] Warning: WARNING: process group has NOT been destroyed before we destruct ProcessGroupNCCL. On normal program exit, the application should call destroy_process_group to ensure that any pending NCCL operations have finished in this process. In rare cases this process can exit before this point and block the progress of another member of the process group. This constraint has always been present,  but this warning has only been added since PyTorch 2.4 (function operator())
Model comparison results saved to ./test_results/20250326_223637/model_comparison_results.txt
```

## ✅ BRO YAY IT'S FKING WORKING

```bash
Llama 3.1 8B + 4 Reward functions
Base Model Accuracy: 0.0938
LoRA Model Accuracy: 0.3125
Improvement: 0.2188

Llama 3.1 8B + 2 reward fucntions
Base Model Accuracy: 0.0625
LoRA Model Accuracy: 0.2188
Improvement: 0.1562
```

- Bro the 1B model suck 👀

```bash
Sample outputs saved to trainer_output_meta-llama_Llama-3.2-1B-Instruct_gpu0_20250327_110154/lora_model_debug_outputs.txt

Evaluation of LoRA model completed
Accuracy: 0.0312
Results saved to model_comparison_results.txt

Model comparison completed.
Base Model Accuracy: 0.0625
LoRA Model Accuracy: 0.0312
Improvement: -0.0312
```
