import os
from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import *

import numpy as np
import torch
import torch.nn as nn
from packaging.version import Version
from trl.trainer.grpo_trainer import (
    Any,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Dataset,
    GenerationConfig,
    GRPOConfig,
    IterableDataset,
    Optional,
    PeftConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    RepeatRandomSampler,
    RewardFunc,
    Sampler,
    SamplingParams,
    SyncRefModelCallback,
    Trainer,
    TrainerCallback,
    Union,
    apply_chat_template,
    broadcast_object_list,
    create_reference_model,
    defaultdict,
    gather,
    gather_object,
    generate_model_card,
    get_comet_experiment_url,
    is_conversational,
    is_deepspeed_zero3_enabled,
    is_peft_model,
    is_wandb_available,
    maybe_apply_chat_template,
    nn,
    os,
    pad,
    prepare_deepspeed,
    set_seed,
    textwrap,
    torch,
    transformers,
    unwrap_model_for_generation,
    version,
    wandb,
)

from config import logger

torch_compile_options = {
    "epilogue_fusion": True,
    "max_autotune": False,
    "shape_padding": True,
    "trace.enabled": False,
    "triton.cudagraphs": False,
}


@torch.compile(
    dynamic=True,
    fullgraph=True,
    options=torch_compile_options,
)
def selective_log_softmax(logits, index):
    logits = logits.to(torch.float32)
    selected_logits = torch.gather(logits, dim=-1, index=index.unsqueeze(-1)).squeeze(-1)
    # loop to reduce peak mem consumption
    # logsumexp_values = torch.stack([torch.logsumexp(lg, dim=-1) for lg in logits])
    logsumexp_values = torch.logsumexp(logits, dim=-1)
    per_token_logps = selected_logits - logsumexp_values  # log_softmax(x_i) = x_i - logsumexp(x)
    return per_token_logps


def grpo_compute_loss(old_logits, new_logits, input_ids, mask, beta, advantages):
    # All Unsloth Zoo code licensed under LGPLv3
    old_logits = old_logits.to(torch.float32)
    new_logits = new_logits.to(torch.float32)

    # Print FULL tensor contents
    # logger.debug("\n🔍 DETAILED TENSOR ANALYSIS:")
    # logger.debug("\n1️⃣ Input IDs:")
    # logger.debug(f"Shape: {input_ids.shape}")
    # Use tensor.cpu().numpy() to safely print content

    # logger.debug(f"Type: {mask.dtype}")
    # logger.debug(f"Sum: {mask.sum().item()}")  # Use .item() to get Python scalar

    # logger.debug("\n3️⃣ Old Logits:")
    # logger.debug(f"Shape: {old_logits.shape}")
    # logger.debug(f"Type: {old_logits.dtype}")
    # logger.debug(f"Mean: {old_logits.mean().item():.4f}")

    # logger.debug("\n4️⃣ New Logits:")
    # logger.debug(f"Shape: {new_logits.shape}")
    # logger.debug(f"Type: {new_logits.dtype}")
    # logger.debug(f"Mean: {new_logits.mean().item():.4f}")

    # logger.debug("\n5️⃣ Advantages:")
    # logger.debug(f"Shape: {advantages.shape}")

    input_ids = input_ids.unsqueeze(-1)

    # x_i - logsumexp(x_i)
    old_x = torch.gather(old_logits, dim=-1, index=input_ids).squeeze(-1)
    new_x = torch.gather(new_logits, dim=-1, index=input_ids).squeeze(-1)
    old = old_x - torch.logsumexp(old_logits, dim=-1)
    new = new_x - torch.logsumexp(new_logits, dim=-1)

    # logger.debug("\n6️⃣ After Gather & LogSumExp:")
    # logger.debug(f"old_x shape: {old_x.shape}, new_x shape: {new_x.shape}")
    # logger.debug(f"old shape: {old.shape}, new shape: {new.shape}")

    # Reverse KL
    kl_i = torch.exp(old - new) - (old - new) - 1.0
    # Full correct reverse KL divergence?? Missing term maybe?
    # kl_i = torch.exp(new) * kl_i

    # Below is forward KL (normal KL)
    # kl_i = torch.exp(old) * (old - new)

    # Must detach - otherwise gradients are not propagated correctly!
    # exp(x - x) == 1
    loss_i = torch.exp(new - new.detach()) * advantages.unsqueeze(1)
    loss_i = -(loss_i - beta * kl_i)

    mask = mask.to(torch.float32)
    n_mask_per_reward = mask.sum(1)

    # See https://github.com/huggingface/trl/pull/2881
    # loss_per_reward = (loss_i * mask).sum(1) / n_mask_per_reward
    # loss = loss_per_reward.mean()

    # Add print statements here for debugging
    # logger.debug(f"🚨 Debug: loss_i shape: {loss_i.shape}")
    # logger.debug(f"🚨 Debug: mask shape: {mask.shape}")

    loss = (loss_i * mask).sum() / mask.sum()

    # Get metrics as well which are folded
    with torch.inference_mode():
        completion_length = n_mask_per_reward.mean()
        mean_kl_per_reward = (kl_i * mask).sum(1) / n_mask_per_reward
        mean_kl = mean_kl_per_reward.mean()
    pass
    return loss, completion_length, mean_kl


class UnslothEfficientGRPO(torch.autograd.Function):
    # All Unsloth Zoo code licensed under LGPLv3
    @staticmethod
    def forward(
        ctx,
        _new_hidden_states,
        _old_hidden_states,
        lm_head,
        _input_ids,
        _mask,
        _advantages,
        beta,
        scaler=None,
        n_chunks=1,
    ):
        def compute_loss(new_hidden_states, old_hidden_states, input_ids, mask, advantages, scaling):
            new_logits = torch.matmul(new_hidden_states, lm_head.t())
            new_logits = new_logits[:, :-1, :]  # exclude the last logit: it corresponds to the next token pred
            old_logits = torch.matmul(old_hidden_states, lm_head.t())
            old_logits = old_logits[:, :-1, :]  # exclude the last logit: it corresponds to the next token pred
            loss, completion_length, mean_kl = grpo_compute_loss(
                old_logits,
                new_logits,
                input_ids,
                mask,
                beta,
                advantages,
            )
            # Scale loss if needed for mixed precision training
            scaled_loss = loss * scaling
            # Must add .loss.detach otherwise autograd uses 2x VRAM
            return scaled_loss, (
                loss.detach(),
                completion_length,
                mean_kl,
            )

        pass

        device = _new_hidden_states.device
        grad_inputs = torch.empty_like(_new_hidden_states)
        accumulated_loss = torch.zeros(1, device=device)
        accumulated_completion_length = torch.zeros(1, device=device)
        accumulated_mean_kl = torch.zeros(1, device=device)

        def accumulate_chunk(new_hidden_states_j, old_hidden_states_j, input_ids_j, mask_j, advantages_j, scaling):
            (
                (chunk_grad_input,),
                (
                    chunk_loss,
                    (
                        unscaled_loss,
                        chunk_completion_length,
                        chunk_mean_kl,
                    ),
                ),
            ) = torch.func.grad_and_value(
                compute_loss,
                argnums=(0,),
                has_aux=True,
            )(new_hidden_states_j, old_hidden_states_j, input_ids_j, mask_j, advantages_j, scaling)
            accumulated_loss.add_(unscaled_loss)
            accumulated_completion_length.add_(chunk_completion_length)
            accumulated_mean_kl.add_(chunk_mean_kl)
            return chunk_grad_input

        pass

        accumulate_chunk = torch.compile(
            accumulate_chunk,
            fullgraph=True,
            options=torch_compile_options,
        )

        grad_inputs_chunks = torch.chunk(grad_inputs, chunks=n_chunks, dim=0)
        new_hidden_states = torch.chunk(_new_hidden_states, chunks=n_chunks, dim=0)
        old_hidden_states = torch.chunk(_old_hidden_states, chunks=n_chunks, dim=0)
        input_ids = torch.chunk(_input_ids, chunks=n_chunks, dim=0)
        mask = torch.chunk(_mask, chunks=n_chunks, dim=0)
        advantages = torch.chunk(_advantages, chunks=n_chunks, dim=0)

        # Get mixed precision scaling if seen
        scaling = scaler.get_scale() if scaler is not None else 1.0

        # Force torch.compile to use dynamic shapes for seqlen dim
        mark_dynamic = lambda x: torch._dynamo.mark_dynamic(x, 1)

        for (
            grad_inputs_j,
            new_hidden_states_j,
            old_hidden_states_j,
            input_ids_j,
            mask_j,
            advantages_j,
        ) in zip(grad_inputs_chunks, new_hidden_states, old_hidden_states, input_ids, mask, advantages):
            mark_dynamic(new_hidden_states_j)
            mark_dynamic(old_hidden_states_j)
            mark_dynamic(input_ids_j)
            mark_dynamic(mask_j)

            grad_inputs_j.copy_(
                accumulate_chunk(new_hidden_states_j, old_hidden_states_j, input_ids_j, mask_j, advantages_j, scaling)
            )
        pass

        grad_inputs.div_(n_chunks)
        accumulated_loss.div_(n_chunks)
        accumulated_completion_length.div_(n_chunks)
        accumulated_mean_kl.div_(n_chunks)
        ctx.save_for_backward(grad_inputs)

        return (
            accumulated_loss,
            accumulated_completion_length,
            accumulated_mean_kl,
        )

    pass

    @staticmethod
    def backward(ctx, grad_output, dcompletion_length, dmean_kl):
        (grad_input,) = ctx.saved_tensors
        return (
            grad_input,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )

    pass


def grpo_accumulated_loss(
    trainer,
    input_ids,
    logits_to_keep,
    completion_mask,
    advantages,
    n_chunks=-1,
):
    # All Unsloth Zoo code licensed under LGPLv3
    bsz, qlen = input_ids.shape
    # Find closest multiple
    factors = [i for i in range(1, bsz + 1) if bsz % i == 0]
    if n_chunks == -1:
        n_chunks = bsz
    n_chunks = factors[min(np.searchsorted(factors, n_chunks), len(factors) - 1)]

    mixed_dtype = torch.float16 if os.environ.get("ACCELERATE_MIXED_PRECISION", "fp16") == "fp16" else torch.bfloat16
    os.environ["UNSLOTH_RETURN_HIDDEN_STATES"] = "1"

    completion_input_ids = input_ids[:, -logits_to_keep:]
    lm_head = trainer.model.get_output_embeddings().weight

    with torch.amp.autocast(device_type="cuda", dtype=mixed_dtype):
        with (
            torch.inference_mode(),
            trainer.accelerator.unwrap_model(trainer.model, keep_fp32_wrapper=False).disable_adapter(),
        ):
            old_hidden_states = trainer.model(input_ids=input_ids, logits_to_keep=logits_to_keep + 1).logits
        pass

        new_hidden_states = trainer.model(input_ids=input_ids, logits_to_keep=logits_to_keep + 1).logits

        loss, completion_length, mean_kl = UnslothEfficientGRPO.apply(
            new_hidden_states,
            old_hidden_states,
            lm_head,
            completion_input_ids,
            completion_mask,
            advantages,
            trainer.beta,
            trainer.accelerator.scaler,
            n_chunks,
        )
        return loss, completion_length, mean_kl

        # Old non efficient code path
        new_logits = torch.matmul(new_hidden_states, lm_head.t())
        new_logits = new_logits[:, :-1, :]  # exclude the last logit: it corresponds to the next token pred
        old_logits = torch.matmul(old_hidden_states, lm_head.t())
        old_logits = old_logits[:, :-1, :]  # exclude the last logit: it corresponds to the next token pred
        loss, completion_length, mean_kl = grpo_compute_loss(
            old_logits,
            new_logits,
            completion_input_ids,
            completion_mask,
            trainer.beta,
            advantages,
        )
        return loss, completion_length, mean_kl
    pass


def vLLMSamplingParams(**kwargs):
    from vllm import SamplingParams

    sampling_params = SamplingParams(**kwargs)
    sampling_params._set_kwargs = kwargs
    return sampling_params


@dataclass
class UnslothGRPOConfig(GRPOConfig):
    """

    Configuration class for the [`GRPOTrainer`].

    Only the parameters specific to GRPO training are listed here. For details on other parameters, refer to the
    [`~transformers.TrainingArguments`] documentation.

    Using [`~transformers.HfArgumentParser`] we can turn this class into
    [argparse](https://docs.python.org/3/library/argparse#module-argparse) arguments that can be specified on the
    command line.

    Parameters:
        > Parameters that control the model and reference model

        model_init_kwargs (`dict[str, Any]` or `None`, *optional*, defaults to `None`):
            Keyword arguments for [`~transformers.AutoModelForCausalLM.from_pretrained`], used when the `model`
            argument of the [`GRPOTrainer`] is provided as a string.

        > Parameters that control the data preprocessing

        remove_unused_columns (`bool`, *optional*, defaults to `False`):
            Whether to only keep the column `"prompt"` in the dataset. If you use a custom reward function that
            requires any column other than `"prompts"` and `"completions"`, you should keep this to `False`.
        max_prompt_length (`int` or `None`, *optional*, defaults to `512`):
            Maximum length of the prompt. If the prompt is longer than this value, it will be truncated left.
        num_generations (`int` or `None`, *optional*, defaults to `8`):
            Number of generations per prompt to sample. The global batch size (num_processes * per_device_batch_size)
            must be divisible by this value.
        temperature (`float`, *optional*, defaults to `0.9`):
            Temperature for sampling. The higher the temperature, the more random the completions.
        max_completion_length (`int` or `None`, *optional*, defaults to `256`):
            Maximum length of the generated completion.
        ds3_gather_for_generation (`bool`, *optional*, defaults to `True`):
            This setting applies to DeepSpeed ZeRO-3. If enabled, the policy model weights are gathered for generation,
            improving generation speed. However, disabling this option allows training models that exceed the VRAM
            capacity of a single GPU, albeit at the cost of slower generation. Disabling this option is not compatible
            with vLLM generation.

        > Parameters that control generation acceleration powered by vLLM

        use_vllm (`bool`, *optional*, defaults to `False`):
            Whether to use vLLM for generating completions. If set to `True`, ensure that a GPU is kept unused for
            training, as vLLM will require one for generation. vLLM must be installed (`pip install vllm`).
        vllm_device (`str`, *optional*, defaults to `"auto"`):
            Device where vLLM generation will run, e.g. `"cuda:1"`. If set to `"auto"` (default), the system will
            automatically select the next available GPU after the last one used for training. This assumes that
            training has not already occupied all available GPUs. If only one device is available, the device will be
            shared between both training and vLLM.
        vllm_gpu_memory_utilization (`float`, *optional*, defaults to `0.9`):
            Ratio (between 0 and 1) of GPU memory to reserve for the model weights, activations, and KV cache on the
            device dedicated to generation powered by vLLM. Higher values will increase the KV cache size and thus
            improve the model's throughput. However, if the value is too high, it may cause out-of-memory (OOM) errors
            during initialization.
        vllm_dtype (`str`, *optional*, defaults to `"auto"`):
            Data type to use for vLLM generation. If set to `"auto"`, the data type will be automatically determined
            based on the model configuration. Find the supported values in the vLLM documentation.
        vllm_max_model_len (`int` or `None`, *optional*, defaults to `None`):
            If set, the `max_model_len` to use for vLLM. This could be useful when running with reduced
            `vllm_gpu_memory_utilization`, leading to a reduced KV cache size. If not set, vLLM will use the model
            context size, which might be much larger than the KV cache, leading to inefficiencies.

        > Parameters that control the training

        learning_rate (`float`, *optional*, defaults to `1e-6`):
            Initial learning rate for [`AdamW`] optimizer. The default value replaces that of
            [`~transformers.TrainingArguments`].
        beta (`float`, *optional*, defaults to `0.04`):
            KL coefficient.
        reward_weights (`list[float]` or `None`, *optional*, defaults to `None`):
            Weights for each reward function. Must match the number of reward functions. If `None`, all rewards are
            weighted equally with weight `1.0`.
        sync_ref_model (`bool`, *optional*, defaults to `False`):
            Whether to synchronize the reference model with the active model every `ref_model_sync_steps` steps, using
            the `ref_model_mixup_alpha` parameter. This synchronization originates from the
            [TR-DPO](https://huggingface.co/papers/2404.09656) paper.
        ref_model_mixup_alpha (`float`, *optional*, defaults to `0.9`):
            α parameter from the [TR-DPO](https://huggingface.co/papers/2404.09656) paper, which controls the mix
            between the current policy and the previous reference policy during updates. The reference policy is
            updated according to the equation: `π_ref = α * π_θ + (1 - α) * π_ref_prev`. To use this parameter, you
            must set `sync_ref_model=True`.
        ref_model_sync_steps (`int`, *optional*, defaults to `64`):
            τ parameter from the [TR-DPO](https://huggingface.co/papers/2404.09656) paper, which determines how
            frequently the current policy is synchronized with the reference policy. To use this parameter, you must
            set `sync_ref_model=True`.

        > Parameters that control the logging

        log_completions (`bool`, *optional*, defaults to `False`):
            Whether to log the completions during training.

    """

    vllm_sampling_params: Optional[Any] = field(
        default=None,
        metadata={"help": "vLLM SamplingParams"},
    )
    unsloth_num_chunks: Optional[int] = field(
        default=-1,
        metadata={"help": "Chunk size to reduce memory usage. -1 is most efficient."},
    )

    def __init__(
        self,
        output_dir=None,
        overwrite_output_dir=None,
        do_train=False,
        do_eval=False,
        do_predict=False,
        eval_strategy="no",
        prediction_loss_only=False,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        per_gpu_train_batch_size=None,
        per_gpu_eval_batch_size=None,
        gradient_accumulation_steps=2,
        eval_accumulation_steps=2,
        eval_delay=0,
        torch_empty_cache_steps=250,
        learning_rate=5e-05,
        weight_decay=0.01,
        adam_beta1=0.9,
        adam_beta2=0.999,
        adam_epsilon=1e-08,
        max_grad_norm=1.0,
        num_train_epochs=3.0,
        max_steps=-1,
        lr_scheduler_type="linear",
        warmup_ratio=0.1,
        warmup_steps=0,
        log_level="passive",
        log_level_replica="warning",
        log_on_each_node=True,
        logging_dir=None,
        logging_strategy="steps",
        logging_first_step=False,
        logging_steps=1,
        logging_nan_inf_filter=False,
        save_strategy="steps",
        save_steps=500,
        save_total_limit=None,
        save_safetensors=True,
        save_on_each_node=False,
        save_only_model=False,
        restore_callback_states_from_checkpoint=False,
        no_cuda=False,
        use_cpu=False,
        use_mps_device=False,
        seed=3407,
        data_seed=3407,
        jit_mode_eval=False,
        use_ipex=False,
        bf16=False,
        fp16=False,
        fp16_opt_level="O1",
        half_precision_backend="auto",
        bf16_full_eval=False,
        fp16_full_eval=False,
        tf32=None,
        local_rank=-1,
        ddp_backend=None,
        tpu_num_cores=None,
        tpu_metrics_debug=False,
        debug="",
        dataloader_drop_last=False,
        eval_steps=None,
        dataloader_num_workers=0,
        dataloader_prefetch_factor=None,
        past_index=-1,
        run_name=None,
        disable_tqdm=None,
        remove_unused_columns=False,
        label_names=None,
        load_best_model_at_end=False,
        metric_for_best_model=None,
        greater_is_better=None,
        ignore_data_skip=False,
        fsdp="",
        fsdp_min_num_params=0,
        fsdp_config=None,
        fsdp_transformer_layer_cls_to_wrap=None,
        accelerator_config=None,
        deepspeed=None,
        label_smoothing_factor=0.0,
        optim="adamw_8bit",
        optim_args=None,
        adafactor=False,
        group_by_length=False,
        length_column_name="length",
        report_to=None,
        ddp_find_unused_parameters=None,
        ddp_bucket_cap_mb=None,
        ddp_broadcast_buffers=None,
        dataloader_pin_memory=True,
        dataloader_persistent_workers=False,
        skip_memory_metrics=True,
        use_legacy_prediction_loop=False,
        push_to_hub=False,
        resume_from_checkpoint=None,
        hub_model_id=None,
        hub_strategy="every_save",
        hub_token=None,
        hub_private_repo=None,
        hub_always_push=False,
        gradient_checkpointing=False,
        gradient_checkpointing_kwargs=None,
        include_inputs_for_metrics=False,
        eval_do_concat_batches=True,
        fp16_backend="auto",
        # evaluation_strategy=None,
        push_to_hub_model_id=None,
        push_to_hub_organization=None,
        push_to_hub_token=None,
        mp_parameters="",
        auto_find_batch_size=False,
        full_determinism=False,
        torchdynamo=None,
        ray_scope="last",
        ddp_timeout=1800,
        torch_compile=False,
        torch_compile_backend=None,
        torch_compile_mode=None,
        dispatch_batches=None,
        split_batches=None,
        include_tokens_per_second=False,
        include_num_input_tokens_seen=False,
        neftune_noise_alpha=None,
        optim_target_modules=None,
        batch_eval_metrics=False,
        eval_on_start=False,
        use_liger_kernel=False,
        eval_use_gather_object=False,
        average_tokens_across_devices=False,
        model_init_kwargs=None,
        max_prompt_length=512,
        num_generations=8,
        temperature=0.9,
        max_completion_length=256,
        ds3_gather_for_generation=True,
        use_vllm=False,
        use_agentic_generate=False,
        vllm_device="auto",
        vllm_gpu_memory_utilization=0.9,
        vllm_dtype="auto",
        vllm_max_model_len=None,
        beta=0.04,
        reward_weights=None,
        sync_ref_model=False,
        ref_model_mixup_alpha=0.9,
        ref_model_sync_steps=64,
        log_completions=False,
        vllm_sampling_params=None,
        unsloth_num_chunks=-1,
        **kwargs,
    ):
        if learning_rate < 1e-7:
            raise FloatingPointError(
                f"Unsloth: Your learning rate of `{learning_rate}` is too small and less than 1e-7! Consider increasing it, otherwise gradient updates will be close to 0!"
            )
        if learning_rate > 1:
            raise OverflowError(
                f"Unsloth: Your learning rate of `{learning_rate}` is way too larger > 1! Consider decreasing it to 1e-1, otherwise gradient updates will explode!"
            )
        if output_dir is None and save_strategy == "steps" and save_steps == 500:
            output_dir = "unsloth_training_checkpoints"
            save_strategy = "no"
        div = per_device_train_batch_size // num_generations
        if div * num_generations != per_device_train_batch_size:
            logger.debug(
                "Unsloth: We now expect `per_device_train_batch_size` to be a multiple of `num_generations`.\nWe will change the batch size of "
                + str(per_device_train_batch_size)
                + " to the `num_generations` of "
                + str(num_generations)
            )
            per_device_train_batch_size = num_generations

        super().__init__(
            output_dir=output_dir,
            overwrite_output_dir=overwrite_output_dir,
            do_train=do_train,
            do_eval=do_eval,
            do_predict=do_predict,
            eval_strategy=eval_strategy,
            prediction_loss_only=prediction_loss_only,
            per_device_train_batch_size=per_device_train_batch_size,
            per_device_eval_batch_size=per_device_eval_batch_size,
            per_gpu_train_batch_size=per_gpu_train_batch_size,
            per_gpu_eval_batch_size=per_gpu_eval_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            eval_accumulation_steps=eval_accumulation_steps,
            eval_delay=eval_delay,
            torch_empty_cache_steps=torch_empty_cache_steps,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            adam_beta1=adam_beta1,
            adam_beta2=adam_beta2,
            adam_epsilon=adam_epsilon,
            max_grad_norm=max_grad_norm,
            num_train_epochs=num_train_epochs,
            max_steps=max_steps,
            lr_scheduler_type=lr_scheduler_type,
            warmup_ratio=warmup_ratio,
            warmup_steps=warmup_steps,
            log_level=log_level,
            log_level_replica=log_level_replica,
            log_on_each_node=log_on_each_node,
            logging_dir=logging_dir,
            logging_strategy=logging_strategy,
            logging_first_step=logging_first_step,
            logging_steps=logging_steps,
            logging_nan_inf_filter=logging_nan_inf_filter,
            save_strategy=save_strategy,
            save_steps=save_steps,
            save_total_limit=save_total_limit,
            save_safetensors=save_safetensors,
            save_on_each_node=save_on_each_node,
            save_only_model=save_only_model,
            restore_callback_states_from_checkpoint=restore_callback_states_from_checkpoint,
            no_cuda=no_cuda,
            use_cpu=use_cpu,
            use_mps_device=use_mps_device,
            seed=seed,
            data_seed=data_seed,
            jit_mode_eval=jit_mode_eval,
            use_ipex=use_ipex,
            bf16=bf16,
            fp16=fp16,
            fp16_opt_level=fp16_opt_level,
            half_precision_backend=half_precision_backend,
            bf16_full_eval=bf16_full_eval,
            fp16_full_eval=fp16_full_eval,
            tf32=tf32,
            local_rank=local_rank,
            ddp_backend=ddp_backend,
            tpu_num_cores=tpu_num_cores,
            tpu_metrics_debug=tpu_metrics_debug,
            debug=debug,
            dataloader_drop_last=dataloader_drop_last,
            eval_steps=eval_steps,
            dataloader_num_workers=dataloader_num_workers,
            dataloader_prefetch_factor=dataloader_prefetch_factor,
            past_index=past_index,
            run_name=run_name,
            disable_tqdm=disable_tqdm,
            remove_unused_columns=remove_unused_columns,
            label_names=label_names,
            load_best_model_at_end=load_best_model_at_end,
            metric_for_best_model=metric_for_best_model,
            greater_is_better=greater_is_better,
            ignore_data_skip=ignore_data_skip,
            fsdp=fsdp,
            fsdp_min_num_params=fsdp_min_num_params,
            fsdp_config=fsdp_config,
            fsdp_transformer_layer_cls_to_wrap=fsdp_transformer_layer_cls_to_wrap,
            accelerator_config=accelerator_config,
            deepspeed=deepspeed,
            label_smoothing_factor=label_smoothing_factor,
            optim=optim,
            optim_args=optim_args,
            adafactor=adafactor,
            group_by_length=group_by_length,
            length_column_name=length_column_name,
            report_to=report_to,
            ddp_find_unused_parameters=ddp_find_unused_parameters,
            ddp_bucket_cap_mb=ddp_bucket_cap_mb,
            ddp_broadcast_buffers=ddp_broadcast_buffers,
            dataloader_pin_memory=dataloader_pin_memory,
            dataloader_persistent_workers=dataloader_persistent_workers,
            skip_memory_metrics=skip_memory_metrics,
            use_legacy_prediction_loop=use_legacy_prediction_loop,
            push_to_hub=push_to_hub,
            resume_from_checkpoint=resume_from_checkpoint,
            hub_model_id=hub_model_id,
            hub_strategy=hub_strategy,
            hub_token=hub_token,
            hub_private_repo=hub_private_repo,
            hub_always_push=hub_always_push,
            gradient_checkpointing=gradient_checkpointing,
            gradient_checkpointing_kwargs=gradient_checkpointing_kwargs,
            include_inputs_for_metrics=include_inputs_for_metrics,
            eval_do_concat_batches=eval_do_concat_batches,
            fp16_backend=fp16_backend,
            # evaluation_strategy=evaluation_strategy,
            push_to_hub_model_id=push_to_hub_model_id,
            push_to_hub_organization=push_to_hub_organization,
            push_to_hub_token=push_to_hub_token,
            mp_parameters=mp_parameters,
            auto_find_batch_size=auto_find_batch_size,
            full_determinism=full_determinism,
            torchdynamo=torchdynamo,
            ray_scope=ray_scope,
            ddp_timeout=ddp_timeout,
            torch_compile=torch_compile,
            torch_compile_backend=torch_compile_backend,
            torch_compile_mode=torch_compile_mode,
            dispatch_batches=dispatch_batches,
            split_batches=split_batches,
            include_tokens_per_second=include_tokens_per_second,
            include_num_input_tokens_seen=include_num_input_tokens_seen,
            neftune_noise_alpha=neftune_noise_alpha,
            optim_target_modules=optim_target_modules,
            batch_eval_metrics=batch_eval_metrics,
            eval_on_start=eval_on_start,
            use_liger_kernel=use_liger_kernel,
            eval_use_gather_object=eval_use_gather_object,
            average_tokens_across_devices=average_tokens_across_devices,
            model_init_kwargs=model_init_kwargs,
            max_prompt_length=max_prompt_length,
            num_generations=num_generations,
            temperature=temperature,
            max_completion_length=max_completion_length,
            ds3_gather_for_generation=ds3_gather_for_generation,
            use_vllm=use_vllm,
            vllm_device=vllm_device,
            vllm_gpu_memory_utilization=vllm_gpu_memory_utilization,
            vllm_dtype=vllm_dtype,
            vllm_max_model_len=vllm_max_model_len,
            beta=beta,
            reward_weights=reward_weights,
            sync_ref_model=sync_ref_model,
            ref_model_mixup_alpha=ref_model_mixup_alpha,
            ref_model_sync_steps=ref_model_sync_steps,
            log_completions=log_completions,
            **kwargs,
        )
        self.vllm_sampling_params = vllm_sampling_params
        self.unsloth_num_chunks = unsloth_num_chunks
        self.use_agentic_generate = use_agentic_generate


pass


class _UnslothGRPOTrainer(Trainer):
    """"""

    _tag_names = ["trl", "grpo"]

    def __init__(
        self,
        model: Union[str, PreTrainedModel],
        reward_funcs: Union[RewardFunc, list[RewardFunc]],
        args: GRPOConfig = None,
        train_dataset: Optional[Union[Dataset, IterableDataset]] = None,
        eval_dataset: Optional[Union[Dataset, IterableDataset, dict[str, Union[Dataset, IterableDataset]]]] = None,
        processing_class: Optional[PreTrainedTokenizerBase] = None,
        reward_processing_classes: Optional[Union[PreTrainedTokenizerBase, list[PreTrainedTokenizerBase]]] = None,
        callbacks: Optional[list[TrainerCallback]] = None,
        optimizers: tuple[Optional[torch.optim.Optimizer], Optional[torch.optim.lr_scheduler.LambdaLR]] = (None, None),
        peft_config: Optional["PeftConfig"] = None,
    ):
        if hasattr(model, "vllm_engine") and hasattr(args, "use_vllm") and (getattr(args, "use_vllm", False) == False):
            args.use_vllm = True
        # Args
        if args is None:
            model_name = model if isinstance(model, str) else model.config._name_or_path
            model_name = model_name.split("/")[-1]
            args = GRPOConfig(f"{model_name}-GRPO")

        # Models
        # Trained model
        model_init_kwargs = args.model_init_kwargs or {}
        if isinstance(model, str):
            model_id = model
            torch_dtype = model_init_kwargs.get("torch_dtype")
            if isinstance(torch_dtype, torch.dtype) or torch_dtype == "auto" or torch_dtype is None:
                pass  # torch_dtype is already a torch.dtype or "auto" or None
            elif isinstance(torch_dtype, str):  # it's a str, but not "auto"
                torch_dtype = getattr(torch, torch_dtype)
                model_init_kwargs["torch_dtype"] = torch_dtype
            else:
                raise ValueError(
                    "Invalid `torch_dtype` passed to `GRPOConfig`. Expected either 'auto' or a string representing "
                    f"a `torch.dtype` (e.g., 'float32'), but got {torch_dtype}."
                )
            # Disable caching if gradient checkpointing is enabled (not supported)
            model_init_kwargs["use_cache"] = (
                False if args.gradient_checkpointing else model_init_kwargs.get("use_cache")
            )
            model = AutoModelForCausalLM.from_pretrained(model, **model_init_kwargs)
        else:
            model_id = model.config._name_or_path
            if args.model_init_kwargs is not None:
                raise ValueError(
                    "You passed `model_init_kwargs` to the `GRPOConfig`, but your model is already instantiated. "
                    "This argument can only be used when the `model` argument is a string."
                )

        if False:
            model = model

        # Reference model
        if is_deepspeed_zero3_enabled():
            self.ref_model = AutoModelForCausalLM.from_pretrained(model_id, **model_init_kwargs)
        elif not is_peft_model(model):
            # If PEFT configuration is not provided, create a reference model based on the initial model.
            self.ref_model = create_reference_model(model)
        else:
            # If PEFT is used, the reference model is not needed since the adapter can be disabled
            # to revert to the initial model.
            self.ref_model = None

        # Processing class
        if processing_class is None:
            processing_class = AutoTokenizer.from_pretrained(model.config._name_or_path, padding_side="left")

        # Reward functions
        if not isinstance(reward_funcs, list):
            reward_funcs = [reward_funcs]
        for i, reward_func in enumerate(reward_funcs):
            if isinstance(reward_func, str):
                reward_funcs[i] = AutoModelForSequenceClassification.from_pretrained(
                    reward_func, num_labels=1, **model_init_kwargs
                )
        self.reward_funcs = reward_funcs

        # Reward weights
        if args.reward_weights is not None:
            if len(args.reward_weights) != len(reward_funcs):
                raise ValueError(
                    f"Number of reward weights ({len(args.reward_weights)}) must match number of reward "
                    f"functions ({len(reward_funcs)})"
                )
            self.reward_weights = torch.tensor(args.reward_weights, dtype=torch.float32)
        else:
            self.reward_weights = torch.ones(len(reward_funcs), dtype=torch.float32)

        # Reward processing class
        if reward_processing_classes is None:
            reward_processing_classes = [None] * len(reward_funcs)
        elif not isinstance(reward_processing_classes, list):
            reward_processing_classes = [reward_processing_classes]
        else:
            if len(reward_processing_classes) != len(reward_funcs):
                raise ValueError("The number of reward processing classes must match the number of reward functions.")

        for i, (reward_processing_class, reward_func) in enumerate(zip(reward_processing_classes, reward_funcs)):
            if isinstance(reward_func, PreTrainedModel):
                if reward_processing_class is None:
                    reward_processing_class = AutoTokenizer.from_pretrained(reward_func.config._name_or_path)
                if reward_processing_class.pad_token_id is None:
                    reward_processing_class.pad_token = reward_processing_class.eos_token
                # The reward model computes the reward for the latest non-padded token in the input sequence.
                # So it's important to set the pad token ID to the padding token ID of the processing class.
                reward_func.config.pad_token_id = reward_processing_class.pad_token_id
                reward_processing_classes[i] = reward_processing_class
        self.reward_processing_classes = reward_processing_classes

        # Data collator
        def data_collator(features):  # No data collation is needed in GRPO
            return features

        # Training arguments
        self.max_prompt_length = args.max_prompt_length
        self.max_completion_length = args.max_completion_length  # = |o_i| in the GRPO paper
        self.num_generations = args.num_generations  # = G in the GRPO paper
        self.use_vllm = args.use_vllm
        self.use_agentic_generate = args.use_agentic_generate

        self.beta = args.beta

        # The trainer estimates the number of FLOPs (floating-point operations) using the number of elements in the
        # input tensor associated with the key "input_ids". However, in GRPO, the sampled data does not include the
        # "input_ids" key. Instead, the available keys is "prompt". As a result, the trainer issues the warning:
        # "Could not estimate the number of tokens of the input, floating-point operations will not be computed." To
        # suppress this warning, we set the "estimate_tokens" key in the model's "warnings_issued" dictionary to True.
        # This acts as a flag to indicate that the warning has already been issued.
        model.warnings_issued["estimate_tokens"] = True

        # Initialize the metrics
        self._metrics = defaultdict(list)
        self.log_completions = args.log_completions

        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            callbacks=callbacks,
            optimizers=optimizers,
        )

        # Check if the per_device_train/eval_batch_size * num processes can be divided by the number of generations
        num_processes = self.accelerator.num_processes
        global_batch_size = args.per_device_train_batch_size * num_processes
        possible_values = [n_gen for n_gen in range(2, global_batch_size + 1) if (global_batch_size) % n_gen == 0]
        if self.num_generations not in possible_values:
            raise ValueError(
                f"The global train batch size ({num_processes} x {args.per_device_train_batch_size}) must be evenly "
                f"divisible by the number of generations per prompt ({self.num_generations}). Given the current train "
                f"batch size, the valid values for the number of generations are: {possible_values}."
            )
        if self.args.eval_strategy != "no":
            global_batch_size = args.per_device_eval_batch_size * num_processes
            possible_values = [n_gen for n_gen in range(2, global_batch_size + 1) if (global_batch_size) % n_gen == 0]
            if self.num_generations not in possible_values:
                raise ValueError(
                    f"The global eval batch size ({num_processes} x {args.per_device_eval_batch_size}) must be evenly "
                    f"divisible by the number of generations per prompt ({self.num_generations}). Given the current "
                    f"eval batch size, the valid values for the number of generations are: {possible_values}."
                )

        # Ensure each process receives a unique seed to prevent duplicate completions when generating with
        # transformers if num_generations exceeds per_device_train_batch_size. We could skip it if we use vLLM, but
        # it's safer to set it in all cases.
        set_seed(args.seed, device_specific=True)

        if self.use_vllm:
            self.llm = model.vllm_engine
            self._last_loaded_step = 0
            self.sampling_params = SamplingParams(
                temperature=args.temperature,
                max_tokens=self.max_completion_length,
                **getattr(getattr(args, "vllm_sampling_params", vLLMSamplingParams()), "_set_kwargs", {}),
            )
        else:
            self.generation_config = GenerationConfig(
                max_new_tokens=self.max_completion_length,
                do_sample=True,
                temperature=args.temperature,
                pad_token_id=processing_class.pad_token_id,
            )

        # Gradient accumulation requires scaled loss. Normally, loss scaling in the parent class depends on whether the
        # model accepts loss-related kwargs. Since we compute our own loss, this check is irrelevant. We set
        # self.model_accepts_loss_kwargs to False to enable scaling.
        self.model_accepts_loss_kwargs = False

        # Add tags to the model
        self.model.add_model_tags(self._tag_names)

        if self.ref_model is not None:
            if self.is_deepspeed_enabled:
                self.ref_model = prepare_deepspeed(self.ref_model, self.accelerator)
            else:
                self.ref_model = self.accelerator.prepare_model(self.ref_model, evaluation_mode=True)

        if args.sync_ref_model:
            self.add_callback(SyncRefModelCallback(ref_model=self.ref_model, accelerator=self.accelerator))

        for i, reward_func in enumerate(self.reward_funcs):
            if isinstance(reward_func, PreTrainedModel):
                self.reward_funcs[i] = self.accelerator.prepare_model(reward_func, evaluation_mode=True)

    def _set_signature_columns_if_needed(self):
        # If `self.args.remove_unused_columns` is True, non-signature columns are removed.
        # By default, this method sets `self._signature_columns` to the model's expected inputs.
        # In GRPOTrainer, we preprocess data, so using the model's signature columns doesn't work.
        # Instead, we set them to the columns expected by the `training_step` method, hence the override.
        if self._signature_columns is None:
            self._signature_columns = ["prompt"]

    def _get_train_sampler(self) -> Sampler:
        # Returns a sampler that ensures each prompt is repeated across multiple processes. This guarantees that
        # identical prompts are distributed to different GPUs, allowing rewards to be computed and normalized correctly
        # within each prompt group. Using the same seed across processes ensures consistent prompt assignment,
        # preventing discrepancies in group formation.
        return RepeatRandomSampler(self.train_dataset, self.num_generations, seed=self.args.seed)

    def _get_eval_sampler(self, eval_dataset) -> Sampler:
        # Returns a sampler that ensures each prompt is repeated across multiple processes. This guarantees that
        # identical prompts are distributed to different GPUs, allowing rewards to be computed and normalized correctly
        # within each prompt group. Using the same seed across processes ensures consistent prompt assignment,
        # preventing discrepancies in group formation.
        return RepeatRandomSampler(eval_dataset, self.num_generations, seed=self.args.seed)

    # Get the per-token log probabilities for the completions for the model and the reference model
    def _get_per_token_logps(self, model, input_ids, attention_mask, logits_to_keep):
        return None  # Unsloth efficient GRPO
        if not hasattr(self, "_autocast_dtype"):
            self._autocast_dtype = (
                torch.float16 if os.environ.get("ACCELERATE_MIXED_PRECISION", "fp16") == "fp16" else torch.bfloat16
            )
        with torch.amp.autocast(device_type="cuda", dtype=self._autocast_dtype):
            # We add 1 to `logits_to_keep` because the last logits of the sequence is later excluded
            logits = model(
                input_ids=input_ids, attention_mask=attention_mask, logits_to_keep=logits_to_keep + 1
            ).logits
            logits = logits[:, :-1, :]  # (B, L-1, V), exclude the last logit: it corresponds to the next token pred

            input_ids = input_ids[:, -logits_to_keep:]
            # For transformers<=4.48, logits_to_keep argument isn't supported, so here we drop logits ourselves.
            # See https://github.com/huggingface/trl/issues/2770
            logits = logits[:, -logits_to_keep:]
            return logits
            # return selective_log_softmax(logits, input_ids)  #  compute logprobs for the input tokens
        pass

    def _move_model_to_vllm(self, *args, **kwargs):
        return None

    def _prepare_inputs(self, inputs: dict[str, Union[torch.Tensor, Any]]) -> dict[str, Union[torch.Tensor, Any]]:
        # logger.debug("\n🔍 DEBUG: Starting _prepare_inputs")
        device = self.accelerator.device
        prompts = [x["prompt"] for x in inputs]
        prompts_text = [maybe_apply_chat_template(example, self.processing_class)["prompt"] for example in inputs]
        # logger.debug("\n1️⃣ Before tokenization:")
        # logger.debug(f"Number of prompts: {len(prompts)}")
        # logger.debug(f"Sample prompt text length: {len(prompts_text[0]) if prompts_text else 0}")

        prompt_inputs = self.processing_class(
            prompts_text, return_tensors="pt", padding=True, padding_side="left", add_special_tokens=False
        )
        prompt_inputs = super()._prepare_inputs(prompt_inputs)
        prompt_ids, prompt_mask = prompt_inputs["input_ids"], prompt_inputs["attention_mask"]

        # logger.debug("\n2️⃣ After initial tokenization:")
        # logger.debug(f"prompt_ids shape: {prompt_ids.shape}")
        # logger.debug(f"prompt_mask shape: {prompt_mask.shape}")
        # logger.debug(f"prompt_mask sum: {prompt_mask.sum().item()}")

        if self.max_prompt_length is not None:
            prompt_ids = prompt_ids[:, -self.max_prompt_length :]
            prompt_mask = prompt_mask[:, -self.max_prompt_length :]
            # logger.debug("\n3️⃣ After prompt length truncation:")
            # logger.debug(f"prompt_ids shape: {prompt_ids.shape}")
            # logger.debug(f"prompt_mask shape: {prompt_mask.shape}")

        # Generate completions using either vLLM or regular generation
        if self.args.use_vllm:
            # First, have main process load weights if needed
            if self.state.global_step != self._last_loaded_step:
                self._move_model_to_vllm()
                self._last_loaded_step = self.state.global_step

            # Generate completions using vLLM: gather all prompts and use them in a single call in the main process
            all_prompts_text = gather_object(prompts_text)
            if self.accelerator.is_main_process:
                # logger.debug(all_prompts_text)
                generate_fn = lambda prompts_text: self.llm.generate(
                    prompts_text,
                    sampling_params=self.sampling_params,
                    use_tqdm=False,
                    lora_request=self.model.load_lora("grpo_trainer_lora_model", load_tensors=True),
                )
                if self.use_agentic_generate:
                    agentic_outputs = self.model.agentic_generate(
                        all_prompts_text,
                        generate_fn,
                    )
                    full_chats = agentic_outputs.full_chat_states
                    final_responses = agentic_outputs.final_response_str
                    # TODO: apply truncation here
                    prompt_inputs = agentic_outputs.prompt_tokens
                    completion_ids = agentic_outputs.response_tokens
                    completion_mask = agentic_outputs.response_masks
                    # for i in range(len(completion_ids)):
                    # logger.debug(f"prompt_inputs {i} len before padding: {len(prompt_inputs[i])}")
                    # logger.debug(f"completion_ids {i} len before padding: {len(completion_ids[i])}")
                    # logger.debug(f"completion_mask {i} len before padding: {len(completion_mask[i])}")

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

                    # for i in range(len(completion_ids)):
                    # logger.debug(f"prompt_inputs {i} len after padding: {len(prompt_inputs[i])}")
                    # logger.debug(f"prompt_ids {i} len after padding: {len(prompt_ids[i])}")
                    # logger.debug(f"completion_mask {i} len after padding: {len(completion_mask[i])}")

                else:
                    outputs = generate_fn(all_prompts_text)
                    completion_ids = [out.token_ids for completions in outputs for out in completions.outputs]
            else:
                completion_ids = [None] * len(all_prompts_text)
            # Broadcast the completions from the main process to all processes, ensuring each process receives its
            # corresponding slice.
            completion_ids = broadcast_object_list(completion_ids, from_process=0)
            process_slice = slice(
                self.accelerator.process_index * len(prompts),
                (self.accelerator.process_index + 1) * len(prompts),
            )
            completion_ids = completion_ids[process_slice]

            # Pad the completions, and concatenate them with the prompts
            completion_ids = [torch.tensor(ids, device=device) for ids in completion_ids]
            # logger.debug("\n4️⃣ Before completion padding:")
            # logger.debug(f"completion_ids shapes: {[ids.shape for ids in completion_ids]}")

            completion_ids = pad(completion_ids, padding_value=self.processing_class.pad_token_id)
            # logger.debug("\n5️⃣ After completion padding:")
            # logger.debug(f"completion_ids shape: {completion_ids.shape}")

            prompt_completion_ids = torch.cat([prompt_ids, completion_ids], dim=1)
            # logger.debug("\n6️⃣ After concatenation:")
            # logger.debug(f"prompt_completion_ids shape: {prompt_completion_ids.shape}")
        else:
            # Regular generation path
            with unwrap_model_for_generation(self.model, self.accelerator) as unwrapped_model:
                prompt_completion_ids = unwrapped_model.generate(
                    prompt_ids, attention_mask=prompt_mask, generation_config=self.generation_config
                )

            # Compute prompt length and extract completion ids
            prompt_length = prompt_ids.size(1)
            prompt_ids = prompt_completion_ids[:, :prompt_length]
            completion_ids = prompt_completion_ids[:, prompt_length:]

        if not self.use_agentic_generate:
            # Mask everything after the first EOS token
            # logger.debug("\n🔍 Starting EOS token detection and masking:")
            # logger.debug(f"completion_ids shape: {completion_ids.shape}")
            # logger.debug(f"eos_token_id: {self.processing_class.eos_token_id}")

            # Debug EOS detection
            is_eos = completion_ids == self.processing_class.eos_token_id
            # logger.debug("\n7️⃣ EOS Detection Details:")
            # logger.debug(f"is_eos shape: {is_eos.shape}")
            # logger.debug(f"Sample is_eos values (first sequence):\n{is_eos[0]}")
            # logger.debug(f"Any EOS tokens found: {is_eos.any().item()}")
            # logger.debug(f"EOS positions: {is_eos.nonzero()}")

            # Debug EOS index tensor creation
            eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
            # logger.debug("\n8️⃣ EOS Index Creation:")
            # logger.debug(f"eos_idx initial shape: {eos_idx.shape}")
            # logger.debug(f"eos_idx initial values: {eos_idx}")

            # Debug the complex indexing operation
            # logger.debug("\n9️⃣ EOS Position Analysis:")
            # logger.debug(f"Sequences with EOS: {is_eos.any(dim=1).sum().item()}")
            # logger.debug(f"First EOS positions: {is_eos.int().argmax(dim=1)}")

            eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
            # logger.debug("\n🔟 After EOS Index Update:")
            # logger.debug(f"Updated eos_idx values: {eos_idx}")

            # Debug sequence indices creation
            sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
            # logger.debug("\n1️⃣1️⃣ Sequence Indices:")
            # logger.debug(f"sequence_indices shape: {sequence_indices.shape}")
            # logger.debug(f"Sample sequence_indices (first row):\n{sequence_indices[0]}")

            # Debug final mask creation
            completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()
            # logger.debug("\n1️⃣2️⃣ Final Completion Mask:")
            # logger.debug(f"completion_mask shape: {completion_mask.shape}")
            # logger.debug(f"Sample mask (first sequence):\n{completion_mask[0]}")
            # logger.debug("Mask statistics:")
            # logger.debug(f"- Total 1s: {completion_mask.sum().item()}")
            # logger.debug(f"- Average sequence length: {completion_mask.sum(dim=1).float().mean().item():.2f}")

            # Add a final validation check
            # logger.debug("\n7️⃣ Final Validation:")
            # logger.debug(f"Input shape: {completion_ids.shape}")
            # logger.debug(f"Mask shape: {completion_mask.shape}")

        # Concatenate prompt_mask with completion_mask for logit computation
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)  # (B*G, P+C)

        logits_to_keep = completion_ids.size(1)  # we only need to compute the logits for the completion tokens

        # this does nothing
        with (
            torch.inference_mode(),
            torch.amp.autocast(
                device_type="cuda",
                dtype=torch.float16
                if os.environ.get("ACCELERATE_MIXED_PRECISION", "fp16") == "fp16"
                else torch.bfloat16,
            )
            if not torch.is_autocast_enabled("cuda")
            else nullcontext(),
        ):
            if self.ref_model is not None:
                ref_per_token_logps = self._get_per_token_logps(
                    self.ref_model, prompt_completion_ids, attention_mask, logits_to_keep
                )
            else:
                with self.accelerator.unwrap_model(self.model, keep_fp32_wrapper=False).disable_adapter():
                    ref_per_token_logps = self._get_per_token_logps(
                        self.model, prompt_completion_ids, attention_mask, logits_to_keep
                    )

        # Decode the generated completions
        if not self.use_agentic_generate:
            completions_text = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)
            if is_conversational(inputs[0]):
                completions = []
                for prompt, completion in zip(prompts, completions_text):
                    bootstrap = prompt.pop()["content"] if prompt[-1]["role"] == "assistant" else ""
                    completions.append([{"role": "assistant", "content": bootstrap + completion}])
            else:
                completions = completions_text
        else:
            completions = full_chats

        rewards_per_func = torch.zeros(len(prompts), len(self.reward_funcs), device=device)
        for i, (reward_func, reward_processing_class) in enumerate(
            zip(self.reward_funcs, self.reward_processing_classes)
        ):
            if isinstance(reward_func, nn.Module):  # Module instead of PretrainedModel for compat with compiled models
                if is_conversational(inputs[0]):
                    messages = [{"messages": p + c} for p, c in zip(prompts, completions)]
                    texts = [apply_chat_template(x, reward_processing_class)["text"] for x in messages]
                else:
                    texts = [p + c for p, c in zip(prompts, completions)]
                reward_inputs = reward_processing_class(
                    texts, return_tensors="pt", padding=True, padding_side="right", add_special_tokens=False
                )
                reward_inputs = super()._prepare_inputs(reward_inputs)
                with (
                    torch.inference_mode(),
                    torch.amp.autocast(
                        device_type="cuda",
                        dtype=torch.float16
                        if os.environ.get("ACCELERATE_MIXED_PRECISION", "fp16") == "fp16"
                        else torch.bfloat16,
                    )
                    if not torch.is_autocast_enabled("cuda")
                    else nullcontext(),
                ):
                    rewards_per_func[:, i] = reward_func(**reward_inputs).logits[:, 0]  # Shape (B*G,)
            else:
                # Repeat all input columns (but "prompt" and "completion") to match the number of generations
                keys = [key for key in inputs[0] if key not in ["prompt", "completion"]]
                reward_kwargs = {key: [example[key] for example in inputs] for key in keys}
                output_reward_func = reward_func(prompts=prompts, completions=completions, **reward_kwargs)
                rewards_per_func[:, i] = torch.tensor(output_reward_func, dtype=torch.float32, device=device)

        # Gather the reward per function: this part is crucial, because the rewards are normalized per group and the
        # completions may be distributed across processes
        rewards_per_func = gather(rewards_per_func)
        # Apply weights to each reward function's output and sum
        rewards = (rewards_per_func * self.reward_weights.to(device).unsqueeze(0)).sum(dim=1)

        # else:
        #     reward_fn = self.reward_funcs[0]
        #     rewards = reward_fn(

        #         prompts=prompts_text,
        #         student_answers=final_responses,
        #         answers=[input["answer"] for input in inputs],
        #     )

        #     rewards = torch.tensor(rewards, dtype=torch.float16, device=device)
        #     rewards_per_func = rewards.unsqueeze(1)

        # Compute grouped-wise rewards
        mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1)
        std_grouped_rewards = rewards.view(-1, self.num_generations).std(dim=1)

        # Normalize the rewards to compute the advantages
        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        std_grouped_rewards = std_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        advantages = (rewards - mean_grouped_rewards) / (std_grouped_rewards + 1e-4)

        # Slice to keep only the local part of the data
        process_slice = slice(
            self.accelerator.process_index * len(prompts),
            (self.accelerator.process_index + 1) * len(prompts),
        )
        advantages = advantages[process_slice]

        # Log the metrics
        reward_per_func = rewards_per_func.mean(0)
        # logger.debug("rewards_per_func:", reward_per_func)
        for i, reward_func in enumerate(self.reward_funcs):
            if isinstance(reward_func, nn.Module):  # Module instead of PretrainedModel for compat with compiled models
                reward_func_name = reward_func.config._name_or_path.split("/")[-1]
            else:
                reward_func_name = reward_func.__name__
            self._metrics[f"rewards/{reward_func_name}"].append(reward_per_func[i].item())

        self._metrics["reward"].append(rewards.mean().item())
        self._metrics["reward_std"].append(std_grouped_rewards.mean().item())
        self._metrics["advantages_mean"].append(advantages.mean().item())
        self._metrics["advantages_std"].append(advantages.std().item())

        if (
            self.log_completions
            and self.state.global_step % self.args.logging_steps == 0
            and "wandb" in self.args.report_to
        ):
            import pandas as pd

            # For logging
            table = {
                "step": [str(self.state.global_step)] * len(rewards),
                "prompt": gather_object(prompts_text),
                "completion": gather_object(completions_text),
                "reward": rewards.tolist(),
            }
            df = pd.DataFrame(table)

            if wandb.run is not None and self.accelerator.is_main_process:
                wandb.log({"completions": wandb.Table(dataframe=df)})

        # Log prompt length
        prompt_length = prompt_mask.sum(dim=1).float().mean().item()
        self._metrics["prompt_length"].append(prompt_length)

        return {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "ref_per_token_logps": ref_per_token_logps,
            "advantages": advantages,
        }

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if return_outputs:
            raise ValueError("The GRPOTrainer does not support returning outputs")
        # Compute the per-token log probabilities for the model

        prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
        completion_ids, completion_mask = inputs["completion_ids"], inputs["completion_mask"]
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        bsz, qlen = input_ids.shape
        # attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        attention_mask = None
        logits_to_keep = completion_ids.size(1)  # we only need to compute the logits for the completion tokens
        _input_ids = input_ids
        _logits_to_keep = logits_to_keep
        per_token_logps = self._get_per_token_logps(model, input_ids, attention_mask, logits_to_keep)

        # Compute the KL divergence between the model and the reference model
        ref_per_token_logps = inputs["ref_per_token_logps"]
        # per_token_kl = torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1

        # x - x.detach() allows for preserving gradients from x
        advantages = inputs["advantages"]
        # per_token_loss = torch.exp(per_token_logps - per_token_logps.detach()) * advantages.unsqueeze(1)
        # per_token_loss = -(per_token_loss - self.beta * per_token_kl)
        # loss = ((per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()
        input_ids = input_ids[:, -logits_to_keep:]
        if False:  # per_token_logps is not None:
            loss, completion_length, mean_kl = grpo_compute_loss(
                ref_per_token_logps,
                per_token_logps,
                input_ids,
                completion_mask,
                self.beta,
                advantages,
            )
        else:
            loss, completion_length, mean_kl = grpo_accumulated_loss(
                self,
                _input_ids,
                logits_to_keep,
                completion_mask,
                advantages,
                n_chunks=self.args.unsloth_num_chunks,
            )

        # Log the metrics
        # completion_length = self.accelerator.gather_for_metrics(completion_mask.sum(1)).float().mean().item()
        self._metrics["completion_length"].append(completion_length.item())

        # mean_kl = ((per_token_kl * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()
        # self._metrics["kl"].append(self.accelerator.gather_for_metrics(mean_kl).mean().item())
        self._metrics["kl"].append(mean_kl.item())
        return loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys: Optional[list[str]] = None):
        inputs = self._prepare_inputs(inputs)
        with torch.no_grad():
            with self.compute_loss_context_manager():
                loss = self.compute_loss(model, inputs)
            loss = loss.mean().detach()
        return loss, None, None

    def log(self, logs: dict[str, float], start_time: Optional[float] = None) -> None:
        metrics = {key: sum(val) / len(val) for key, val in self._metrics.items()}  # average the metrics

        # This method can be called both in training and evaluation. When called in evaluation, the keys in `logs`
        # start with "eval_". We need to add the prefix "eval_" to the keys in `metrics` to match the format.
        if next(iter(logs.keys())).startswith("eval_"):
            metrics = {f"eval_{key}": val for key, val in metrics.items()}

        logs = {**logs, **metrics}
        if version.parse(transformers.__version__) >= version.parse("4.47.0.dev0"):
            super().log(logs, start_time)
        else:  # transformers<=4.46
            super().log(logs)
        self._metrics.clear()

    def create_model_card(
        self,
        model_name: Optional[str] = None,
        dataset_name: Optional[str] = None,
        tags: Union[str, list[str], None] = None,
    ):
        """
        Creates a draft of a model card using the information available to the `Trainer`.

        Args:
            model_name (`str` or `None`, *optional*, defaults to `None`):
                Name of the model.
            dataset_name (`str` or `None`, *optional*, defaults to `None`):
                Name of the dataset used for training.
            tags (`str`, `list[str]` or `None`, *optional*, defaults to `None`):
                Tags to be associated with the model card.
        """
        if not self.is_world_process_zero():
            return

        if hasattr(self.model.config, "_name_or_path") and not os.path.isdir(self.model.config._name_or_path):
            base_model = self.model.config._name_or_path
        else:
            base_model = None

        tags = tags or []
        if isinstance(tags, str):
            tags = [tags]

        if hasattr(self.model.config, "unsloth_version"):
            tags.append("unsloth")

        citation = textwrap.dedent(
            """\
            @article{zhihong2024deepseekmath,
                title        = {{DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models}},
                author       = {Zhihong Shao and Peiyi Wang and Qihao Zhu and Runxin Xu and Junxiao Song and Mingchuan Zhang and Y. K. Li and Y. Wu and Daya Guo},
                year         = 2024,
                eprint       = {arXiv:2402.03300},
            }
            """
        )

        model_card = generate_model_card(
            base_model=base_model,
            model_name=model_name,
            hub_model_id=self.hub_model_id,
            dataset_name=dataset_name,
            tags=tags,
            wandb_url=wandb.run.get_url() if is_wandb_available() and wandb.run is not None else None,
            comet_url=get_comet_experiment_url(),
            trainer_name="GRPO",
            trainer_citation=citation,
            paper_title="DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models",
            paper_id="2402.03300",
        )

        model_card.save(os.path.join(self.args.output_dir, "README.md"))


class UnslothGRPOTrainer(_UnslothGRPOTrainer):
    """

    Trainer for the Group Relative Policy Optimization (GRPO) method. This algorithm was initially proposed in the
    paper [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://huggingface.co/papers/2402.03300).

    Example:

    ```python
    from datasets import load_dataset
    from trl import GRPOTrainer

    dataset = load_dataset("trl-lib/tldr", split="train")

    def reward_func(completions, **kwargs):
        # Dummy reward function that rewards completions with more unique letters.
        return [float(len(set(completion))) for completion in completions]

    trainer = GRPOTrainer(
        model="Qwen/Qwen2-0.5B-Instruct",
        reward_funcs=reward_func,
        train_dataset=dataset,
    )

    trainer.train()
    ```

    Args:
        model (`Union[str, PreTrainedModel]`):
            Model to be trained. Can be either:

            - A string, being the *model id* of a pretrained model hosted inside a model repo on huggingface.co, or
              a path to a *directory* containing model weights saved using
              [`~transformers.PreTrainedModel.save_pretrained`], e.g., `'./my_model_directory/'`. The model is
              loaded using [`~transformers.AutoModelForCausalLM.from_pretrained`] with the keywork arguments
              in `args.model_init_kwargs`.
            - A [`~transformers.PreTrainedModel`] object. Only causal language models are supported.
        reward_funcs (`Union[RewardFunc, list[RewardFunc]]`):
            Reward functions to be used for computing the rewards. To compute the rewards, we call all the reward
            functions with the prompts and completions and sum the rewards. Can be either:

            - A single reward function, such as:
                - A string: The *model ID* of a pretrained model hosted inside a model repo on huggingface.co, or a
                path to a *directory* containing model weights saved using
                [`~transformers.PreTrainedModel.save_pretrained`], e.g., `'./my_model_directory/'`. The model is loaded
                using [`~transformers.AutoModelForSequenceClassification.from_pretrained`] with `num_labels=1` and the
                keyword arguments in `args.model_init_kwargs`.
                - A [`~transformers.PreTrainedModel`] object: Only sequence classification models are supported.
                - A custom reward function: The function is provided with the prompts and the generated completions,
                  plus any additional columns in the dataset. It should return a list of rewards. For more details, see
                  [Using a custom reward function](#using-a-custom-reward-function).
            - A list of reward functions, where each item can independently be any of the above types. Mixing different
            types within the list (e.g., a string model ID and a custom reward function) is allowed.
        args ([`GRPOConfig`], *optional*, defaults to `None`):
            Configuration for this trainer. If `None`, a default configuration is used.
        train_dataset ([`~datasets.Dataset`] or [`~datasets.IterableDataset`]):
            Dataset to use for training. It must include a column `"prompt"`. Any additional columns in the dataset is
            ignored. The format of the samples can be either:

            - [Standard](dataset_formats#standard): Each sample contains plain text.
            - [Conversational](dataset_formats#conversational): Each sample contains structured messages (e.g., role
              and content).
        eval_dataset ([`~datasets.Dataset`], [`~datasets.IterableDataset`] or `dict[str, Union[Dataset, IterableDataset]]`):
            Dataset to use for evaluation. It must meet the same requirements as `train_dataset`.
        processing_class ([`~transformers.PreTrainedTokenizerBase`], *optional*, defaults to `None`):
            Processing class used to process the data. The padding side must be set to "left". If `None`, the
            processing class is loaded from the model's name with [`~transformers.AutoTokenizer.from_pretrained`].
        reward_processing_classes (`Union[PreTrainedTokenizerBase, list[PreTrainedTokenizerBase]]`, *optional*, defaults to `None`):
            Processing classes corresponding to the reward functions specified in `reward_funcs`. Can be either:

            - A single processing class: Used when `reward_funcs` contains only one reward function.
            - A list of processing classes: Must match the order and length of the reward functions in `reward_funcs`.
            If set to `None`, or if an element of the list corresponding to a [`~transformers.PreTrainedModel`] is
            `None`, the tokenizer for the model is automatically loaded using [`~transformers.AutoTokenizer.from_pretrained`].
            For elements in `reward_funcs` that are custom reward functions (not [`~transformers.PreTrainedModel`]),
            the corresponding entries in `reward_processing_classes` are ignored.
        callbacks (list of [`~transformers.TrainerCallback`], *optional*, defaults to `None`):
            List of callbacks to customize the training loop. Will add those to the list of default callbacks
            detailed in [here](https://huggingface.co/docs/transformers/main_classes/callback).

            If you want to remove one of the default callbacks used, use the [`~transformers.Trainer.remove_callback`]
            method.
        optimizers (`tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]`, *optional*, defaults to `(None, None)`):
            A tuple containing the optimizer and the scheduler to use. Will default to an instance of [`AdamW`] on your
            model and a scheduler given by [`get_linear_schedule_with_warmup`] controlled by `args`.
        peft_config ([`~peft.PeftConfig`], *optional*, defaults to `None`):
            PEFT configuration used to wrap the model. If `None`, the model is not wrapped.

    """

    def __init__(
        self,
        model,
        reward_funcs,
        args=None,
        train_dataset=None,
        eval_dataset=None,
        processing_class=None,
        reward_processing_classes=None,
        callbacks=None,
        peft_config=None,
        **kwargs,
    ):
        if args is None:
            args = UnslothGRPOConfig()
        use_bf16 = getattr(args, "bf16", False)
        use_fp16 = getattr(args, "fp16", False)
        dtype = getattr(model.config, "torch_dtype", None)
        if dtype is None:
            dtype = model.get_input_embeddings().dtype
        from unsloth_zoo.utils import _get_dtype

        dtype = _get_dtype(dtype)
        float16 = dtype == torch.float16
        if float16 and use_bf16:
            raise TypeError(
                "Unsloth: Model is in float16 precision but you want to use bfloat16 precision. Set fp16 to `True` and bf16 to `False`"
            )
        if not float16 and use_fp16:
            raise TypeError(
                "Unsloth: Model is in bfloat16 precision but you want to use float16 precision. Set fp16 to `False` and bf16 to `True`"
            )
        if not use_bf16 and not use_fp16:
            args.fp16 = float16
            args.bf16 = not float16
            os.environ["ACCELERATE_MIXED_PRECISION"] = "fp16" if float16 else "bf16"
        if getattr(args, "eval_dataset", None) is not None and getattr(args, "eval_strategy", "no") == "no":
            args.eval_strategy = "steps"
            if getattr(args, "eval_steps", None) is None:
                args.eval_steps = 0.1
        ga_steps = getattr(args, "gradient_accumulation_steps", None)
        if ga_steps is not None and ga_steps > 1:
            from transformers import __version__ as transformers_version

            if Version(transformers_version) <= Version("4.45.2"):
                print(
                    "**** Unsloth: Please use our fixed gradient_accumulation_steps by updating transformers, TRL and Unsloth!\n"
                    "`pip install --upgrade --no-cache-dir --force-reinstall --no-deps unsloth transformers trl unsloth_zoo`"
                )
        if getattr(args, "eval_strategy", "no") != "no":
            eval_bsz = getattr(args, "per_device_eval_batch_size", 8)
            if eval_bsz == 8 and args.per_device_train_batch_size < eval_bsz:
                args.per_device_eval_batch_size = args.per_device_train_batch_size
            if getattr(args, "eval_accumulation_steps", None) is None and ga_steps is not None:
                args.eval_accumulation_steps = ga_steps
        fp16_full_eval = getattr(args, "fp16_full_eval", False)
        bf16_full_eval = getattr(args, "bf16_full_eval", False)
        if args.fp16 and bf16_full_eval:
            args.bf16_full_eval = False
            args.fp16_full_eval = True
        if args.bf16 and fp16_full_eval:
            args.bf16_full_eval = True
            args.fp16_full_eval = False
        if not bf16_full_eval and not fp16_full_eval:
            args.bf16_full_eval = args.bf16
            args.fp16_full_eval = args.fp16
        if "max_seq_length" not in locals() and not hasattr(args, "max_seq_length"):
            pass
        else:
            model_max_seq_length = getattr(model, "max_seq_length", None)
            args_max_seq_length = getattr(args, "max_seq_length", None)
            if args_max_seq_length is None and model_max_seq_length is not None:
                max_seq_length = model.max_seq_length
                if hasattr(args, "max_seq_length"):
                    args.max_seq_length = max_seq_length
        if model is not None and hasattr(model, "for_training"):
            model.for_training()
        if "tokenizer" in locals() and hasattr(tokenizer, "padding_side"):
            tokenizer.padding_side = "right"
        if "processing_class" in locals():
            if hasattr(processing_class, "padding_side"):
                processing_class.padding_side = "right"
            if hasattr(processing_class, "tokenizer") and hasattr(processing_class.tokenizer, "padding_side"):
                processing_class.tokenizer.padding_side = "right"
        other_metrics = []
        if not isinstance(reward_funcs, list):
            _reward_funcs = [reward_funcs]
        else:
            _reward_funcs = reward_funcs
        for reward_func in _reward_funcs:
            try:
                reward_func_name = reward_func.__name__
                other_metrics.append(f"rewards/{reward_func_name}")
            except:
                pass

        from unsloth_zoo.logging_utils import PatchRLStatistics

        PatchRLStatistics("grpo_trainer", other_metrics)

        super().__init__(
            model=model,
            reward_funcs=reward_funcs,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            reward_processing_classes=reward_processing_classes,
            callbacks=callbacks,
            peft_config=peft_config,
            **kwargs,
        )
        if hasattr(self, "neftune_hook_handle"):
            self.neftune_hook_handle.remove()
            if hasattr(self, "neftune_hook_handle"):
                del self.neftune_hook_handle
        if getattr(args, "neftune_noise_alpha", None) is not None:
            model.get_input_embeddings().neftune_noise_alpha = self.neftune_noise_alpha
        pass


pass
