import os

from unsloth import FastLanguageModel, is_bfloat16_supported

import src.UnslothGRPOTrainerTemp as UnslothGRPOTrainerTemp
from src.config import (
    MODEL_CONFIG,
    MODEL_NAME,
    OUTPUT_DIR,
    TRAINING_CONFIG,
    get_sampling_params,
    init_training_dirs,
    logger,
    update_log_path,
)

# Import reward functions
from src.rl_helpers import (
    build_reward_correctness_fn,
    get_qa_dataset,
    reward_exact_match_chunk_query,
    reward_formatting,
    reward_retry_behavior,
    run_agent,
)

# Initialize training directories
paths = init_training_dirs()

# Update logger to use the training directory
update_log_path(paths["log_dir"])
logger.info(f"Training output directory: {paths['output_dir']}")
logger.info(f"Logs are being saved to both ./logs and {paths['log_dir']}")

# Initialize model and tokenizer
logger.info(f"Initializing model {MODEL_NAME}")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=MODEL_CONFIG["max_seq_length"],
    load_in_4bit=True,  # False for LoRA 16bit
    fast_inference=True,  # Enable vLLM fast inference
    max_lora_rank=MODEL_CONFIG["lora_rank"],
    gpu_memory_utilization=MODEL_CONFIG["gpu_memory_utilization"],
)

# Setup LoRA
logger.info("Setting up LoRA adapter")
model = FastLanguageModel.get_peft_model(
    model,
    r=MODEL_CONFIG["lora_rank"],
    target_modules=MODEL_CONFIG["target_modules"],
    lora_alpha=MODEL_CONFIG["lora_rank"],
    use_gradient_checkpointing=True,  # Enable long context finetuning
    random_state=3407,
)

# Load datasets
logger.info("Loading datasets")
train_dataset, test_dataset = get_qa_dataset()
logger.info(f"Loaded {len(train_dataset)} training examples and {len(test_dataset)} test examples")

# Setup training arguments
logger.info("Setting up training arguments")
training_args = UnslothGRPOTrainerTemp.UnslothGRPOConfig(
    use_vllm=True,  # use vLLM for fast inference!
    use_agentic_generate=True,  # use agentic generation
    **TRAINING_CONFIG,
    bf16=is_bfloat16_supported(),
    fp16=not is_bfloat16_supported(),
    output_dir=OUTPUT_DIR,
    # report_to="tensorboard",  # ❓ Does't have billions of tensorboard files if set report to right here
)


# Setup model generation functions
def agentic_generate(
    prompts: list,
    generate_fn,
    max_generations: int = 10,
):
    return run_agent(generate_fn, tokenizer, prompts, max_generations)


model.agentic_generate = agentic_generate

# Setup verifier
logger.info("Setting up verifier")
verifier_sampling_params = get_sampling_params(temperature=0.1)


def verifier_generate_fn(inputs):
    return model.fast_generate(
        inputs,
        sampling_params=verifier_sampling_params,
    )


# Setup trainer
logger.info("Initializing trainer")
trainer = UnslothGRPOTrainerTemp.UnslothGRPOTrainer(
    model=model,
    processing_class=tokenizer,
    reward_funcs=[
        build_reward_correctness_fn(
            verifier_generate_fn,
            tokenizer,
            log_file=os.path.join(paths["log_dir"], "qa_log.txt"),
        ),
        reward_formatting,
        reward_retry_behavior,
        reward_exact_match_chunk_query,
    ],
    args=training_args,
    train_dataset=train_dataset,
)

# Train the model
if __name__ == "__main__":
    logger.info("Starting training")
    trainer.train()
    logger.info("Training completed")
    logger.info(f"Model saved to {OUTPUT_DIR}")
