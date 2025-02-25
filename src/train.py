# train.py
import torch.distributed as dist
import atexit
import sys
import os
import wandb

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models import load_model
from src.trainer import create_trainer

def cleanup_process_group():
    if dist.is_initialized():
        dist.destroy_process_group()

def main():

    # start a new wandb run to track this script
    wandb.init(
        # set the wandb entity where your project will be logged (generally your team name)
        # entity="my-awesome-team-name",

        # set the wandb project where this run will be logged
        project="rlaif",

        # track hyperparameters and run metadata
        config={}
    )

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
