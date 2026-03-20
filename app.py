import streamlit as st
import torch
import os
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

st.set_page_config(page_title="LLM Tool-Calling Demo", layout="wide")

st.title("🛠️ LLM Tool-Calling Demo (LoRA & DPO)")
st.markdown("Test the baseline model vs the SFT model vs the DPO-aligned model to see how well they output reliable JSON tool calls.")

BASE_MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
SFT_ADAPTER = "results/sft_model"
DPO_ADAPTER = "results/dpo_model"
MERGED_SFT = "results/merged_sft_model"

@st.cache_resource
def load_tokenizer():
    tz = AutoTokenizer.from_pretrained(BASE_MODEL_NAME, trust_remote_code=True)
    if tz.pad_token is None:
        tz.pad_token = tz.eos_token
    return tz

@st.cache_resource
def load_model(model_choice):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32 if device == "cpu" else torch.bfloat16
    
    # We clear cache to free memory when switching models
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if model_choice == "Base Model":
        return AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_NAME, device_map=device, torch_dtype=dtype
        )
    elif model_choice == "SFT Model":
        if not os.path.exists(SFT_ADAPTER):
            return None
        base = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_NAME, device_map=device, torch_dtype=dtype
        )
        return PeftModel.from_pretrained(base, SFT_ADAPTER)
    elif model_choice == "DPO Model":
        if not os.path.exists(DPO_ADAPTER) or not os.path.exists(MERGED_SFT):
            return None
        # DPO was trained on the merged SFT model
        base = AutoModelForCausalLM.from_pretrained(
            MERGED_SFT, device_map=device, torch_dtype=dtype
        )
        return PeftModel.from_pretrained(base, DPO_ADAPTER)

tokenizer = load_tokenizer()

SYSTEM_PROMPT = """You are a highly capable AI assistant equipped with tools. Your task is to analyze the user's request and output ONLY a valid JSON object representing a tool call. Do not add any conversational text.
Available tools:
1. `get_weather(location: str)` - Gets current weather for a location.
2. `calculate(expression: str)` - Evaluates a mathematical expression."""

# Sidebar
st.sidebar.header("Settings")
model_type = st.sidebar.selectbox(
    "Select Model Version",
    ["Base Model", "SFT Model", "DPO Model"]
)

# Load selected model
model = load_model(model_type)

if model is None:
    st.error(f"Error: {model_type} files not found. Did you run the training script?")
else:
    st.success(f"{model_type} loaded successfully!")

    # Main UI
    user_prompt = st.text_input("Enter your request:", placeholder="e.g., What's the weather in Tokyo?")
    
    if st.button("Generate Tool Call"):
        if not user_prompt:
            st.warning("Please enter a prompt.")
        else:
            with st.spinner("Generating response..."):
                messages = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt}
                ]
                
                input_ids = tokenizer.apply_chat_template(
                    messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
                ).to(model.device)
                
                with torch.no_grad():
                    outputs = model.generate(
                        input_ids, 
                        max_new_tokens=100, 
                        temperature=0.1, 
                        do_sample=False,
                        pad_token_id=tokenizer.eos_token_id
                    )
                
                # Extract only generation
                generated_ids = outputs[0][len(input_ids[0]):]
                response = tokenizer.decode(generated_ids, skip_special_tokens=True)
                
                # Output parsing
                st.subheader("Model Raw Output:")
                st.text(response)
                
                # Attempt to parse as JSON
                try:
                    clean_str = response.replace("```json", "").replace("```", "").strip()
                    parsed_json = json.loads(clean_str)
                    st.subheader("Extracted JSON / Tool Call:")
                    st.json(parsed_json)
                    st.success("Successfully generated valid JSON format!")
                except json.JSONDecodeError:
                    st.error("Failed to parse output as strictly formatted JSON. Model hallucinated or failed to format correctly.")
