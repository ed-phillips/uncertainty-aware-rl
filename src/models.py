# src/models.py
from unsloth import FastLanguageModel, PatchFastRL

# Apply the patch before anything else is imported or instantiated.
PatchFastRL("GRPO", FastLanguageModel)

def load_model(model_name: str, max_seq_length: int = 1024, lora_rank: int = 64):
    from unsloth import is_bfloat16_supported  # local import if needed
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        load_in_4bit=True,
        fast_inference=True,
        max_lora_rank=lora_rank,
        gpu_memory_utilization=0.5,
    )
    
    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_rank,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_alpha=lora_rank,
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )
    return model, tokenizer
