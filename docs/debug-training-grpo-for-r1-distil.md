# Debug training grpo for r1 distil

- I want to be able to continue to finetune the model from r1 distil checkpoints
- The errors also occurred in normal Qwen 2.5 1.5B Instruct
- The root cause is that the mask and the ids have different length, which is caused by custom mask logic only made for llama architecture.

## Debug strategy

Debugging Strategy:
The goal is to ensure that for every chat state i, the length of response_toks[i] is exactly the same as the length of response_masks[i] after all processing (slicing and truncation) within the final loop of run_agent.

## FOUND IT

```python
                        print(f" prompt_inputs {i} len before padding: {len(prompt_inputs[i])}")
                        print(f" completion_ids {i} len before padding: {len(completion_ids[i])}")
                        print(f" completion_mask {i} len before padding: {len(completion_mask[i])}")
                    prompt_ids = pad(
                        prompt_inputs,
                        padding_value=self.processing_class.pad_token_id,
                        padding_side="left",
                    ).to(device)
                    completion_mask = pad(
                        completion_mask,
                        padding_value=0,
                        padding_side="right",
                    ).to(device)
                    # print length after padding
                    for i in range(len(prompt_inputs)):
                        print(f" prompt_ids {i} len after padding: {len(prompt_ids[i])}")
                        print(f" completion_ids {i} len after padding: {len(completion_ids[i])}")
                        print(f" completion_mask {i} len after padding: {len(completion_mask[i])}")
```

- Deepseek R1 (the pattern is mask = id + 2, then magically turn into 1025?)

```bash
 prompt_inputs 0 len before padding: 214
 completion_ids 0 len before padding: 99
 completion_mask 0 len before padding: 101
 prompt_inputs 1 len before padding: 214
 completion_ids 1 len before padding: 312
 completion_mask 1 len before padding: 314
 prompt_inputs 2 len before padding: 214
 completion_ids 2 len before padding: 296
 completion_mask 2 len before padding: 298
 prompt_inputs 3 len before padding: 214
 completion_ids 3 len before padding: 270
 completion_mask 3 len before padding: 272
 prompt_inputs 4 len before padding: 214
 completion_ids 4 len before padding: 1024
 completion_mask 4 len before padding: 1025
 prompt_inputs 5 len before padding: 214
 completion_ids 5 len before padding: 71
 completion_mask 5 len before padding: 72
 prompt_inputs 6 len before padding: 214
 completion_ids 6 len before padding: 76
 completion_mask 6 len before padding: 78
 prompt_inputs 7 len before padding: 214
 completion_ids 7 len before padding: 1024
 completion_mask 7 len before padding: 1025
 prompt_ids 0 len after padding: 214
 completion_ids 0 len after padding: 99
 completion_mask 0 len after padding: 1025
 prompt_ids 1 len after padding: 214
 completion_ids 1 len after padding: 312
 completion_mask 1 len after padding: 1025
 prompt_ids 2 len after padding: 214
 completion_ids 2 len after padding: 296
 completion_mask 2 len after padding: 1025
 prompt_ids 3 len after padding: 214
 completion_ids 3 len after padding: 270
 completion_mask 3 len after padding: 1025
 prompt_ids 4 len after padding: 214
 completion_ids 4 len after padding: 1024
 completion_mask 4 len after padding: 1025
 prompt_ids 5 len after padding: 214
 completion_ids 5 len after padding: 71
 completion_mask 5 len after padding: 1025
 prompt_ids 6 len after padding: 214
 completion_ids 6 len after padding: 76
 completion_mask 6 len after padding: 1025
 prompt_ids 7 len after padding: 214
 completion_ids 7 len after padding: 1024
 completion_mask 7 len after padding: 1025
```

- and this is llama

```bash
 prompt_inputs 0 len before padding: 240
 completion_ids 0 len before padding: 572
 completion_mask 0 len before padding: 572
 prompt_inputs 1 len before padding: 240
 completion_ids 1 len before padding: 323
 completion_mask 1 len before padding: 323
 prompt_inputs 2 len before padding: 240
 completion_ids 2 len before padding: 58
 completion_mask 2 len before padding: 58
 prompt_inputs 3 len before padding: 240
 completion_ids 3 len before padding: 61
 completion_mask 3 len before padding: 61
 prompt_inputs 4 len before padding: 240
 completion_ids 4 len before padding: 292
 completion_mask 4 len before padding: 292
 prompt_inputs 5 len before padding: 240
 completion_ids 5 len before padding: 588
 completion_mask 5 len before padding: 588
 prompt_inputs 6 len before padding: 240
 completion_ids 6 len before padding: 617
 completion_mask 6 len before padding: 617
 prompt_inputs 7 len before padding: 240
 completion_ids 7 len before padding: 62
 completion_mask 7 len before padding: 62
 prompt_ids 0 len after padding: 240
 completion_ids 0 len after padding: 572
 completion_mask 0 len after padding: 617
 prompt_ids 1 len after padding: 240
 completion_ids 1 len after padding: 323
 completion_mask 1 len after padding: 617
 prompt_ids 2 len after padding: 240
 completion_ids 2 len after padding: 58
 completion_mask 2 len after padding: 617
 prompt_ids 3 len after padding: 240
 completion_ids 3 len after padding: 61
 completion_mask 3 len after padding: 617
 prompt_ids 4 len after padding: 240
 completion_ids 4 len after padding: 292
 completion_mask 4 len after padding: 617
 prompt_ids 5 len after padding: 240
 completion_ids 5 len after padding: 588
 completion_mask 5 len after padding: 617
 prompt_ids 6 len after padding: 240
 completion_ids 6 len after padding: 617
 completion_mask 6 len after padding: 617
 prompt_ids 7 len after padding: 240
 completion_ids 7 len after padding: 62
 completion_mask 7 len after padding: 617
```

## Bug summarise

```bash
The immediate cause of the crash (TorchRuntimeError) was that the mask tensor had a different sequence length dimension (e.g., 574) than the loss_i tensor (e.g., 294) it was being multiplied with element-wise inside the loss calculation.
You can't multiply tensors of shape (B, SeqLen1) and (B, SeqLen2) element-wise if SeqLen1 is not equal to SeqLen2. The fix ensures both tensors have the same sequence length before the multiplication happens.
```

```bash
What Happened: The code crashed with a TorchRuntimeError indicating a shape mismatch during tensor multiplication (loss_i * mask) inside the grpo_compute_loss function, specifically when running under torch.compile.

The Core Issue: The completion_mask tensor (representing which completion tokens are valid) was being passed into the loss calculation with a sequence length (e.g., 574) that reflected the initial length of the generated sequence before final processing or slicing. However, the loss_i tensor (representing the per-token loss contribution) was correctly calculated based on the intended completion length (logits_to_keep, e.g., 294).
```

## The Error

```bash
Search results: []
2025-04-01 13:06:42 | DEBUG    | src.rl_helpers_r1_distil:reward_exact_match_chunk_query:745 - Reward for prompt 7: 0.0
2025-04-01 13:06:42 | INFO     | src.rl_helpers_r1_distil:reward_exact_match_chunk_query:781 - Chunk Query Rewards Summary:
2025-04-01 13:06:42 | INFO     | src.rl_helpers_r1_distil:reward_exact_match_chunk_query:782 - Total prompts: 8
2025-04-01 13:06:42 | INFO     | src.rl_helpers_r1_distil:reward_exact_match_chunk_query:783 - Correct matches: 2.0
2025-04-01 13:06:42 | INFO     | src.rl_helpers_r1_distil:reward_exact_match_chunk_query:784 - Average reward: 0.250
2025-04-01 13:06:42 | INFO     | src.rl_helpers_r1_distil:reward_exact_match_chunk_query:785 - Reward std: 0.433
rewards_per_func: tensor([0.6250, 0.4375, 0.9500, 0.2500], device='cuda:0')
2025-04-01 13:06:43 | CRITICAL | src.config:exception_handler:132 - Unhandled exception
Traceback (most recent call last):

> File "/home/thinhlpg/code/DeepSearch/train_grpo_r1_distil.py", line 125, in <module>
    trainer.train()
    │       └ <function Trainer.train at 0x7d71f573b560>
    └ <src.UnslothGRPOTrainerTemp.UnslothGRPOTrainer object at 0x7d71982cde10>

...

    raise error_type(message_evaluated)
          │          └ 'The size of tensor a (s4) must match the size of tensor b (s7) at non-singleton dimension 1)'
          └ <class 'RuntimeError'>

torch._dynamo.exc.TorchRuntimeError: Failed running call_function <built-in function mul>(*(GradTrackingTensor(lvl=1, value=
    FakeTensor(..., device='cuda:0', size=(1, s4))
), GradTrackingTensor(lvl=1, value=
    FakeTensor(..., device='cuda:0', size=(1, s7))
)), **{}):
The size of tensor a (s4) must match the size of tensor b (s7) at non-singleton dimension 1)

from user code:
   File "/home/thinhlpg/code/DeepSearch/src/UnslothGRPOTrainerTemp.py", line 186, in accumulate_chunk
    ) = torch.func.grad_and_value(
  File "/home/thinhlpg/miniconda3/envs/deepsearch-py311/lib/python3.11/site-packages/torch/_functorch/apis.py", line 442, in wrapper
    return eager_transforms.grad_and_value_impl(
  File "/home/thinhlpg/miniconda3/envs/deepsearch-py311/lib/python3.11/site-packages/torch/_functorch/vmap.py", line 48, in fn
    return f(*args, **kwargs)
  File "/home/thinhlpg/miniconda3/envs/deepsearch-py311/lib/python3.11/site-packages/torch/_functorch/eager_transforms.py", line 1407, in grad_and_value_impl
    output = func(*args, **kwargs)
  File "/home/thinhlpg/code/DeepSearch/src/UnslothGRPOTrainerTemp.py", line 143, in compute_loss
    loss, completion_length, mean_kl = grpo_compute_loss(
  File "/home/thinhlpg/code/DeepSearch/src/UnslothGRPOTrainerTemp.py", line 112, in grpo_compute_loss
    loss = (loss_i * mask).sum() / mask.sum()
```
