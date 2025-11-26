from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from trl import DPOTrainer, DPOConfig
import torch

model_name = "/workspace/models/Qwen2.5-3B-Instruct"
dataset_path = "/workspace/repo/web3_dpo_dataset.jsonl"

ds = load_dataset("json", data_files=dataset_path, split="train")

tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
tokenizer.pad_token = tokenizer.eos_token

bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb,
    device_map="auto"
)

config = DPOConfig(
    output_dir="/workspace/checkpoints",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=5e-6,
    num_train_epochs=3,
    logging_steps=5,
    save_strategy="epoch",
    bf16=True
)

trainer = DPOTrainer(
    model=model,
    ref_model=None,
    processing_class=tokenizer,
    args=config,
    beta=0.1,
    train_dataset=ds
)

trainer.train()
trainer.save_model("/workspace/checkpoints/final")
tokenizer.save_pretrained("/workspace/checkpoints/final")
print("Training complete")

