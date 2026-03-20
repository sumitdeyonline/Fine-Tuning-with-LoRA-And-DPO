import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, PeftModel
from trl import DPOTrainer

def train_dpo():
    base_model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    sft_adapter_path = "results/sft_model"
    dataset_path = "data/dpo_dataset.jsonl"
    output_dir = "results/dpo_model"

    if not os.path.exists(sft_adapter_path):
        print(f"SFT adapter not found at {sft_adapter_path}. Please run train_sft.py first.")
        return

    # Quantization limits
    device_map = "auto"
    quant_config = None
    if torch.cuda.is_available():
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

    # 1. Load the base model
    print("Loading base model and applying SFT adapter for merging...")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        quantization_config=quant_config,
        device_map=device_map,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32
    )

    # 2. Load the SFT adapter
    # Note: For bitsandbytes quantization, merging might not be fully supported directly on 4bit,
    # but for simplicity, PEFT DPOTrainer with load_in_4bit can manage if we just pass peft_config.
    # Actually, DPOTrainer handles everything nicely if we just load the SFT Model as the active model with a new peft config,
    # OR we merge the SFT model first. If we can't merge due to 4bit, we'll train DPO directly on the base model with SFT data implicitly, or we just load it.
    # To avoid 4-bit merging issues on varied hardware, let's load base model without 4bit, merge, then reload.
    
    del model # Release Memory
    
    # Reload Base Without 4-bit to Merge SFT Adapter safely
    base_model_fp16 = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        device_map="cpu", # Merge on CPU safely
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True
    )
    
    # Load SFT adapter and merge
    merged_model = PeftModel.from_pretrained(base_model_fp16, sft_adapter_path)
    merged_model = merged_model.merge_and_unload()
    print("Merged SFT model into base. Now setting this as the base for DPO.")
    
    # Save the merged model temporarily to disk so we can load it properly with 4-bit
    merged_model_path = "results/merged_sft_model"
    merged_model.save_pretrained(merged_model_path)
    
    del base_model_fp16
    del merged_model
    import gc
    gc.collect()
    
    # 3. Load the Merged Model (Reference Model) and the Active Model for DPO
    model = AutoModelForCausalLM.from_pretrained(
        merged_model_path,
        quantization_config=quant_config,
        device_map=device_map,
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32
    )

    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    tokenizer.save_pretrained(merged_model_path)

    # Load DPO Dataset
    dataset = load_dataset("json", data_files=dataset_path, split="train")
    
    def process_dpo_dataset(example):
        # Format 'prompt', 'chosen', and 'rejected' using chat template
        prompt = tokenizer.apply_chat_template(example["prompt"], tokenize=False, add_generation_prompt=True)
        chosen = tokenizer.apply_chat_template(example["prompt"] + example["chosen"], tokenize=False)[len(prompt):]
        rejected = tokenizer.apply_chat_template(example["prompt"] + example["rejected"], tokenize=False)[len(prompt):]
        
        return {
            "prompt": prompt,
            "chosen": chosen,
            "rejected": rejected,
        }
        
    dpo_dataset = dataset.map(process_dpo_dataset, remove_columns=dataset.column_names)

    # PEFT Config for DPO
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
        task_type="CAUSAL_LM"
    )

    # Training Arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        optim="adamw_torch" if not torch.cuda.is_available() else "paged_adamw_8bit",
        learning_rate=5e-5,
        save_steps=50,
        logging_steps=10,
        max_steps=50, # Demo steps
        warmup_ratio=0.1,
        fp16=False,
        bf16=torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False,
        report_to="none"
    )

    # DPO Trainer
    # TRL DPOTrainer automatically treats `model` as the active model with `peft_config` applied.
    # Since we don't provide `ref_model`, it dynamically creates one or disables adapter for reference forward pass.
    dpo_trainer = DPOTrainer(
        model,
        ref_model=None, # Peft handles reference inherently when peft_config is supplied
        args=training_args,
        beta=0.1,
        train_dataset=dpo_dataset,
        tokenizer=tokenizer,
        peft_config=peft_config,
    )

    print("Starting DPO Training...")
    dpo_trainer.train()

    # Save Final Model
    dpo_trainer.model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Final DPO model saved to {output_dir}")

if __name__ == "__main__":
    train_dpo()
