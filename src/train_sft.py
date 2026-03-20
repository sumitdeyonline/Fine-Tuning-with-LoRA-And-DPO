import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer

def train_sft():
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    dataset_path = "data/sft_dataset.jsonl"
    output_dir = "results/sft_model"

    # Load Dataset
    dataset = load_dataset("json", data_files=dataset_path, split="train")

    # Quantization Config (QLoRA)
    # Check if CUDA is available, as bitsandbytes 4-bit requires CUDA on most setups.
    # If not, we fall back to standard bfloat16 or float16 if supported.
    device_map = "auto"
    quant_config = None
    
    if torch.cuda.is_available():
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    else:
        print("CUDA not detected. Skipping bitsandbytes 4-bit quantization. Training might be slow or OOM.")

    # Load Model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quant_config,
        device_map=device_map,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32
    )

    if quant_config:
        model = prepare_model_for_kbit_training(model)

    # Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # LoRA Config
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
        task_type="CAUSAL_LM"
    )

    model = get_peft_model(model, peft_config)

    # Convert the messages column to the format expected by SFTTrainer if the model tokenizer has a chat template
    def formatting_prompts_func(example):
        output_texts = []
        for i in range(len(example['messages'])):
            text = tokenizer.apply_chat_template(example['messages'][i], tokenize=False, add_generation_prompt=False)
            output_texts.append(text)
        return output_texts

    # Training Arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        optim="adamw_torch" if not torch.cuda.is_available() else "paged_adamw_8bit",
        save_steps=50,
        logging_steps=10,
        learning_rate=2e-4,
        weight_decay=0.001,
        fp16=False,
        bf16=torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False,
        max_grad_norm=0.3,
        max_steps=100, # Set to a low number for demonstration. For full training, use num_train_epochs=3
        warmup_ratio=0.03,
        group_by_length=True,
        lr_scheduler_type="cosine",
        report_to="none" # Disable wandb for local
    )

    # Initialize SFTTrainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=peft_config,
        formatting_func=formatting_prompts_func,
        max_seq_length=512,
        tokenizer=tokenizer,
        args=training_args,
    )

    # Start Training
    print("Starting SFT Training...")
    trainer.train()

    # Save Model
    trainer.model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"SFT model saved to {output_dir}")

if __name__ == "__main__":
    train_sft()
