# src/trainer.py
from trl import GRPOConfig, GRPOTrainer
from unsloth import is_bfloat16_supported
from src.data_loader import get_gsm8k_questions
from src.rewards import (xmlcount_reward_func, soft_format_reward_func, 
                         strict_format_reward_func, int_reward_func, 
                         correctness_reward_func)

def create_trainer(model, tokenizer):
    dataset = get_gsm8k_questions()
    
    training_args = GRPOConfig(
        use_vllm=True,
        learning_rate=5e-6,
        adam_beta1=0.9,
        adam_beta2=0.99,
        weight_decay=0.1,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        optim="adamw_8bit",
        logging_steps=1,
        bf16=is_bfloat16_supported(),
        fp16=not is_bfloat16_supported(),
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        num_generations=8,
        max_prompt_length=256,
        max_completion_length=200,
        max_steps=250,
        save_steps=250,
        max_grad_norm=0.1,
        report_to="none",
        output_dir="outputs",
    )
    
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[
            xmlcount_reward_func,
            soft_format_reward_func,
            strict_format_reward_func,
            int_reward_func,
            correctness_reward_func,
        ],
        args=training_args,
        train_dataset=dataset,
    )
    
    return trainer
