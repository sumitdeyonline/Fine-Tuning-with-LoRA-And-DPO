import os
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

SYSTEM_PROMPT = """You are a highly capable AI assistant equipped with tools. Your task is to analyze the user's request and output ONLY a valid JSON object representing a tool call. Do not add any conversational text.
Available tools:
1. `get_weather(location: str)` - Gets current weather for a location.
2. `calculate(expression: str)` - Evaluates a mathematical expression."""

TEST_PROMPTS = [
    ("What's the weather in Tokyo?", "get_weather"),
    ("Calculate 12 * 12", "calculate"),
    ("Tell me the weather for Berlin.", "get_weather"),
    ("What is 100 / 4?", "calculate"),
    ("Is it raining in London right now?", "get_weather")
]

def check_json_validity(output_str, expected_tool):
    """Parses output and checks if it's valid JSON for the expected tool."""
    try:
        # Sometimes models output Markdown code blocks, strip them
        clean_str = output_str.replace("```json", "").replace("```", "").strip()
        data = json.loads(clean_str)
        if data.get("name") == expected_tool and "arguments" in data:
            return True, data
        return False, data
    except json.JSONDecodeError:
        return False, None

def evaluate(model, tokenizer, name):
    print(f"\n--- Evaluating {name} ---")
    correct = 0
    total = len(TEST_PROMPTS)
    
    for prompt_text, expected_tool in TEST_PROMPTS:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt_text}
        ]
        
        input_ids = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
        ).to(model.device)
        
        # Determine max length and stop criteria
        with torch.no_grad():
            outputs = model.generate(
                input_ids, 
                max_new_tokens=64, 
                temperature=0.1, 
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
            
        # Extract only the newly generated tokens
        generated_ids = outputs[0][len(input_ids[0]):]
        response = tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        is_valid, parsed_data = check_json_validity(response, expected_tool)
        if is_valid:
            correct += 1
        
        print(f"Prompt: {prompt_text}")
        print(f"Raw Output: {response}")
        print(f"Valid Format: {is_valid}")
        print("-" * 20)
        
    accuracy = (correct / total) * 100
    print(f"[{name}] Accuracy: {accuracy:.2f}% ({correct}/{total})")
    return accuracy

def main():
    base_model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    sft_adapter_path = "results/sft_model"
    dpo_model_path = "results/dpo_model"
    merged_sft_path = "results/merged_sft_model"

    print("Loading base model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Base Model
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        device_map=device,
        torch_dtype=torch.float32 if device == "cpu" else torch.bfloat16
    )
    
    base_acc = evaluate(base_model, tokenizer, "Base Model")
    del base_model
    import gc; gc.collect()
    
    # SFT Model
    sft_acc = 0.0
    if os.path.exists(sft_adapter_path):
        print("\nLoading SFT Model...")
        sft_base = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            device_map=device,
            torch_dtype=torch.float32 if device == "cpu" else torch.bfloat16
        )
        sft_model = PeftModel.from_pretrained(sft_base, sft_adapter_path)
        sft_acc = evaluate(sft_model, tokenizer, "SFT Model (QLoRA)")
        del sft_model; del sft_base; gc.collect()
    else:
        print("\nSFT model not found, skipping.")

    # DPO Model
    dpo_acc = 0.0
    if os.path.exists(dpo_model_path) and os.path.exists(merged_sft_path):
        print("\nLoading DPO Model...")
        dpo_base = AutoModelForCausalLM.from_pretrained(
            merged_sft_path, # DPO was trained on top of merged SFT
            device_map=device,
            torch_dtype=torch.float32 if device == "cpu" else torch.bfloat16
        )
        dpo_model = PeftModel.from_pretrained(dpo_base, dpo_model_path)
        dpo_acc = evaluate(dpo_model, tokenizer, "DPO Model")
        del dpo_model; del dpo_base; gc.collect()
    else:
        print("\nDPO model not found, skipping.")

    print("\n" + "="*40)
    print("FINAL METRICS (Strict JSON Tool-Calling Format)")
    print("="*40)
    print(f"Base Model Accuracy: {base_acc:.2f}%")
    print(f"SFT Model Accuracy:  {sft_acc:.2f}%")
    print(f"DPO Model Accuracy:  {dpo_acc:.2f}%")

if __name__ == "__main__":
    main()
