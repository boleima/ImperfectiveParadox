import argparse
import json
import os
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Dict, Any, Optional

# --- Prompt Templates ---

PROMPT_TEMPLATES = {
    "zero-shot": {
        "system": """System Prompt: You are a strict logician. Your task is to determine if a Hypothesis is necessarily true given a Premise.
If the Hypothesis MUST be true based only on the Premise, output "True".
If the Hypothesis is contradicted by the Premise, output "False".
If the Hypothesis might be true but is not explicitly guaranteed by the Premise, output "Unknown".
Do not use common sense assumptions. Only use the text provided.""",
        "user_format": """User Input: Premise: {premise} Hypothesis: {hypothesis}
Please respond with ONLY one of the following options: "True", "False", or "Unknown"."""
    },
    "dap": {
        "system": """System Prompt: You are a strict logician. Your task is to determine if a Hypothesis is necessarily true given a Premise.
If the Hypothesis MUST be true based only on the Premise, output "True".
If the Hypothesis is contradicted by the Premise, output "False".
If the Hypothesis might be true but is not explicitly guaranteed by the Premise, output "Unknown".

Important Linguistic Rule: Note the distinction between activity and accomplishment verbs.
- For accomplishments/goal-oriented actions (e.g., "was building", "was writing"), the progressive form does NOT imply completion.
- For activities (e.g., "was running", "was singing"), the process implies the action occurred.
Do not use common sense assumptions. Only use the text provided.""",
        "user_format": """User Input: Premise: {premise} Hypothesis: {hypothesis}
Please respond with ONLY one of the following options: "True", "False", or "Unknown"."""
    },
    "cot": {
        "system": """System Prompt: You are a strict logician. Your task is to determine if a Hypothesis is necessarily true given a Premise.
If the Hypothesis MUST be true based only on the Premise, output "True".
If the Hypothesis is contradicted by the Premise, output "False".
If the Hypothesis might be true but is not explicitly guaranteed by the Premise, output "Unknown".
Do not use common sense assumptions. Only use the text provided.""",
        "user_format": """Premise: {premise}
Hypothesis: {hypothesis}

Instruction: First, analyze the temporal status of the event in the premise. Does the action have a defined endpoint? Was it completed? Then, provide your final label (True, False, or Unknown).

Output Requirement: You must output your response in strict JSON format containing exactly two keys:
1. "reasoning": A string explaining your step-by-step logic.
2. "label": The final judgment ("True", "False", or "Unknown").

Do not include markdown formatting (like ```json). Return ONLY the JSON object.
Example Format:
{{
  "reasoning": "The premise describes a process...",
  "label": "True" / "False" / "Unknown"
}}"""
    },
"counterfactual": {
        "system": """System Prompt: You are a strict logician. Your task is to determine if a Hypothesis is necessarily true given a Premise.
If the Hypothesis MUST be true based only on the Premise, output "True".
If the Hypothesis is contradicted by the Premise, output "False".
If the Hypothesis might be true but is not explicitly guaranteed by the Premise, output "Unknown".
Do not use common sense assumptions. Only use the text provided.""",
        "user_format": """Premise: {premise}
Hypothesis: {hypothesis}

Instruction: 
1. First, list 3 possible real-world scenarios where the action in the premise occurs but the result in the hypothesis is NOT achieved (e.g., interruptions, failures).
2. Based on these possibilities, determine if the hypothesis is necessarily true.

Output Requirement: You must output your response in strict JSON format containing exactly two keys:
1. "possible_interruptions": a list of three strings for 3 possible real-world scenarios.
2. "label": The final judgment ("True", "False", or "Unknown").

Do not include markdown formatting (like ```json). Return ONLY the JSON object.
Example Format:
{{
  "possible_interruptions": ["..."],
  "label": "True" / "False" / "Unknown"
}}"""
    },
}

# --- Helper Functions ---

def load_model_and_tokenizer(model_name_or_path: str):
    print(f"Loading model from: {model_name_or_path}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype="auto",
            device_map="auto",
            trust_remote_code=True,
        )
        return model, tokenizer
    except Exception as e:
        print(f"Error loading model {model_name_or_path}: {e}")
        return None, None

def prepare_prompt(tokenizer, premise: str, hypothesis: str, prompt_type: str = "simple") -> str:
    """Constructs the prompt based on the selected template style."""
    template = PROMPT_TEMPLATES.get(prompt_type)
    if not template:
        raise ValueError(f"Prompt type '{prompt_type}' not found. Available: {list(PROMPT_TEMPLATES.keys())}")

    system_content = template["system"]
    user_content = template["user_format"].format(premise=premise, hypothesis=hypothesis)

    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content}
    ]

    # Apply chat template if available (Standard for modern Instruct/Chat models)
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except Exception:
            pass # Fallback if template application fails
    
    # Fallback concatenation
    return f"{system_content}\n\n{user_content}"

def generate_response(model, tokenizer, text: str, max_new_tokens: int = 256):
    device = next(model.parameters()).device
    model_inputs = tokenizer(text, return_tensors="pt").to(device)

    generated_ids = model.generate(
        input_ids=model_inputs["input_ids"],
        attention_mask=model_inputs.get("attention_mask", None),
        max_new_tokens=max_new_tokens,
        do_sample=False, # Greedy decoding for reproducibility
        pad_token_id=tokenizer.eos_token_id
    )

    # Extract only the new tokens
    input_len = model_inputs["input_ids"].shape[1]
    new_tokens = generated_ids[0][input_len:]
    
    if len(new_tokens) == 0:
        return ""

    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

def process_dataset(input_file: str, output_file: str, model, tokenizer, prompt_type: str, max_new_tokens: int):
    if not os.path.exists(input_file):
        print(f"Error: Input file {input_file} not found.")
        return

    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    results = []
    print(f"Processing {len(data)} items using prompt style: '{prompt_type}'...")

    for item in tqdm(data, desc="Inference"):
        premise = item.get("premise", "")
        hypothesis = item.get("hypothesis", "")
        
        # Construct Prompt
        prompt_text = prepare_prompt(tokenizer, premise, hypothesis, prompt_type)
        
        # Generate
        try:
            prediction = generate_response(model, tokenizer, prompt_text, max_new_tokens)
        except Exception as e:
            prediction = f"<generation_error: {str(e)}>"
        
        # Store result
        # We copy the original item to keep metadata, and add the prediction
        result_item = item.copy()
        result_item["prompt_used"] = prompt_text
        result_item["prediction"] = prediction
        result_item["model_prompt_type"] = prompt_type
        results.append(result_item)

    # Save
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as fout:
        json.dump(results, fout, ensure_ascii=False, indent=4)
    
    print(f"✅ Results saved to {output_file}")

# --- Main Execution ---

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run NLI Inference with different models and prompts.")
    
    parser.add_argument("--model_path", type=str, required=True, help="Path to local model or HuggingFace model ID")
    parser.add_argument("--input_file", type=str, default="data/imperfectiveNLI.json", help="Path to input JSON file")
    parser.add_argument("--output_dir", type=str, default="results", help="Directory to save results")
    parser.add_argument("--prompt_type", type=str, default="simple", choices=list(PROMPT_TEMPLATES.keys()), help="Type of prompt strategy to use")
    parser.add_argument("--max_new_tokens", type=int, default=512, help="Max tokens to generate")
    
    args = parser.parse_args()

    # Determine Output Filename automatically based on model name and prompt
    model_name_clean = args.model_path.strip("/").split("/")[-1]
    output_filename = f"{model_name_clean}_{args.prompt_type}.json"
    output_file_path = os.path.join(args.output_dir, output_filename)

    # Load
    model, tokenizer = load_model_and_tokenizer(args.model_path)

    if model and tokenizer:
        # Run
        process_dataset(
            input_file=args.input_file,
            output_file=output_file_path,
            model=model,
            tokenizer=tokenizer,
            prompt_type=args.prompt_type,
            max_new_tokens=args.max_new_tokens
        )
