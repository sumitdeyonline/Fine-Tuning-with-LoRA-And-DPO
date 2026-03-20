import json
import random
import os

# Create data directory if it doesn't exist
os.makedirs("data", exist_ok=True)

# Templates for prompts
WEATHER_PROMPTS = [
    "What's the weather like in {city}?",
    "Tell me the weather for {city}.",
    "Is it raining in {city} right now?",
    "Give me the current weather forecast for {city}",
    "I need to know the temperature in {city}"
]

CALC_PROMPTS = [
    "Calculate {expression}",
    "What is {expression}?",
    "Math time: {expression}",
    "Can you compute {expression} for me?",
    "Help me with this math: {expression}"
]

CITIES = ["London", "New York", "Paris", "Tokyo", "Berlin", "Sydney", "Mumbai", "San Francisco"]
EXPRESSIONS = ["2 + 2", "5 * 10", "144 / 12", "3.14 * 5^2", "100 - 45", "9 * 9", "1024 / 2"]

SYSTEM_PROMPT = """You are a highly capable AI assistant equipped with tools. Your task is to analyze the user's request and output ONLY a valid JSON object representing a tool call. Do not add any conversational text.
Available tools:
1. `get_weather(location: str)` - Gets current weather for a location.
2. `calculate(expression: str)` - Evaluates a mathematical expression."""

def generate_sft_data(num_samples=200):
    dataset = []
    for _ in range(num_samples):
        task_type = random.choice(["weather", "calc"])
        if task_type == "weather":
            city = random.choice(CITIES)
            prompt = random.choice(WEATHER_PROMPTS).format(city=city)
            completion = json.dumps({"name": "get_weather", "arguments": {"location": city}})
        else:
            exp = random.choice(EXPRESSIONS)
            prompt = random.choice(CALC_PROMPTS).format(expression=exp)
            completion = json.dumps({"name": "calculate", "arguments": {"expression": exp}})
        
        # Standard ChatML format
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": completion}
        ]
        
        dataset.append({"messages": messages})
        
    with open("data/sft_dataset.jsonl", "w") as f:
        for item in dataset:
            f.write(json.dumps(item) + "\n")
    print(f"Generated {num_samples} SFT samples in data/sft_dataset.jsonl")

def generate_dpo_data(num_samples=200):
    dataset = []
    for _ in range(num_samples):
        task_type = random.choice(["weather", "calc"])
        
        # We need chosen (perfect) and rejected (bad or conversational)
        if task_type == "weather":
            city = random.choice(CITIES)
            prompt = random.choice(WEATHER_PROMPTS).format(city=city)
            chosen = json.dumps({"name": "get_weather", "arguments": {"location": city}})
            
            # Create a rejected response (e.g., hallucinated conversational text before JSON, or wrong tool, or malformed JSON)
            rejected_variants = [
                f"Sure! Here is the weather tool for you: {chosen}",
                json.dumps({"name": "weather_api", "arguments": {"city": city}}), # Wrong tool name / arg
                f'{{name: get_weather, arguments: {{location: "{city}"}}}}' # Invalid JSON (missing quotes)
            ]
            rejected = random.choice(rejected_variants)
            
        else:
            exp = random.choice(EXPRESSIONS)
            prompt = random.choice(CALC_PROMPTS).format(expression=exp)
            chosen = json.dumps({"name": "calculate", "arguments": {"expression": exp}})
            
            rejected_variants = [
                f"The answer is computing... {chosen}",
                json.dumps({"name": "math", "arguments": {"eq": exp}}), # Wrong tool name
                f'{{"name": "calculate", "arguments": "{exp}"}}' # Wrong argument format
            ]
            rejected = random.choice(rejected_variants)
            
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ]
        
        dataset.append({
            "prompt": messages, # DPO in trl standard format is often prompt (list of dicts) + chosen (list of dicts) + rejected (list of dicts)
            "chosen": [{"role": "assistant", "content": chosen}],
            "rejected": [{"role": "assistant", "content": rejected}]
        })

    with open("data/dpo_dataset.jsonl", "w") as f:
        for item in dataset:
            f.write(json.dumps(item) + "\n")
    print(f"Generated {num_samples} DPO samples in data/dpo_dataset.jsonl")

if __name__ == "__main__":
    random.seed(42)
    generate_sft_data(500)
    generate_dpo_data(500)
