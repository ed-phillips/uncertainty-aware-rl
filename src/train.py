# train.py
import torch.distributed as dist
import atexit
from src.models import load_model
from src.trainer import create_trainer

def cleanup_process_group():
    if dist.is_initialized():
        dist.destroy_process_group()

def main():
    model_name = "Qwen/Qwen2.5-3B-Instruct"
    model, tokenizer = load_model(model_name)
    
    trainer = create_trainer(model, tokenizer)
    trainer.train()
    
    # Save the LoRA adapter after training
    model.save_lora("grpo_saved_lora")

if __name__ == "__main__":
    try:
        main()
    finally:
        cleanup_process_group()
