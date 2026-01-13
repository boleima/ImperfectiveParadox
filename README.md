# Imperfective Paradox

This repository provides a streamlined framework for evaluating Large Language Models (LLMs) on the **Imperfective Paradox**, using a Natural Language Inference (NLI) task. It is designed to test logical deduction capabilities, specifically focusing on the **imperfective aspect** (determining if an action was completed) for two types of actions Activity and Accomplishment. We provide a diagnostic dataset, **ImperfectiveNLI**, and an inference code to conduct the experiment.

Check our paper here: 

## ✨ Features

- **Multi-Strategy Prompting**: Easily switch between different reasoning techniques without changing code:
  - **`zero-shot`**: Standard Zero-Shot classification (True/False/Unknown).
  - **`dap`**: Injects specific linguistic rules (Activity vs. Accomplishment verbs) into the system prompt.
  - **`cot`**: Chain-of-Thought approach focusing on the temporal endpoints of actions.
  - **`counterfactual`**: Counterfactual approach asking the model to list potential real-world interruptions first, forcing the model to think about the interruptions.
- **Model Agnostic**: Compatible with any Hugging Face Transformer model (Llama 3, Mistral, Qwen, DeepSeek, etc.).
- **Robust formatting**: Automatically applies the correct `chat_template` for instruction-tuned models.

## 📂 Data Format
```json
[
  {
    "id": "A_001",
    "group": "A_Interrupted_Accomplishment",
    "verb_class": "Creation",
    "verb": "build",
    "premise": "The carpenter was building a gazebo, but a storm destroyed the frame before the roof was on.",
    "hypothesis": "The carpenter built a gazebo.",
    "label": "False",
  },
  ...
]
```
## 🛠️ Run
```bash
python run_inference.py \
    --model_path "meta-llama/Llama-3.1-8B-Instruct" \
    --input_file "data/imperfectiveNLI.json" \
    --output_dir "results" \
    --prompt_type "zero-shot"
```
