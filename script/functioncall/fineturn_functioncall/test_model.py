# test_qwen.py
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

import torch

from datasets import load_dataset
from trl import SFTConfig, SFTTrainer
from peft import LoraConfig, get_peft_model, TaskType
import torch
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, 
    TrainingArguments, Trainer,set_seed,
    DataCollatorForLanguageModeling
)

base_model_path = "/data/postgraduates/2024/chenjiarui/Model/Qwen/Qwen3-1.7B"
lora_checkpoint = "/data/postgraduates/2024/chenjiarui/Model/Agent/script/fineturn_functioncall/qwen_functioncall_full_finetune/checkpoint-400"
device = "cuda" if torch.cuda.is_available() else "cpu"

# -----------------------------
tokenizer = AutoTokenizer.from_pretrained(
    lora_checkpoint,
    trust_remote_code=True
)

from peft import AutoPeftModelForCausalLM

# model = AutoPeftModelForCausalLM.from_pretrained(
#     lora_checkpoint,
#     low_cpu_mem_usage=True,
#     device_map="auto",
# )


merged_model = AutoModelForCausalLM.from_pretrained(
    lora_checkpoint,
    low_cpu_mem_usage=True,
    device_map="auto",
)

# Merge LoRA and base model
# merged_model = model.merge_and_unload()

prompt="""
You are a function calling AI model. You are provided with function signatures within <tools></tools> XML tags.You may call one or more functions to assist with the user query. Don't make assumptions about what values to plug into functions.Here are the available tools:<tools> [{'type': 'function', 'function': {'name': 'generate_random_number', 'description': 'Generate a random number within a range', 'parameters': {'type': 'object', 'properties': {'min': {'type': 'integer', 'description': 'The minimum value'}, 'max': {'type': 'integer', 'description': 'The maximum value'}}, 'required': ['min', 'max']}}}, {'type': 'function', 'function': {'name': 'calculate_discount', 'description': 'Calculate the discounted price', 'parameters': {'type': 'object', 'properties': {'original_price': {'type': 'number', 'description': 'The original price'}, 'discount_percentage': {'type': 'number', 'description': 'The percentage of discount'}}, 'required': ['original_price', 'discount_percentage']}}}] </tools>Use the following pydantic model json schema for each tool call you will make: {'title': 'FunctionCall', 'type': 'object', 'properties': {'arguments': {'title': 'Arguments', 'type': 'object'}, 'name': {'title': 'Name', 'type': 'string'}}, 'required': ['arguments', 'name']}For each function call return a json object with function name and arguments within <tool_call></tool_call> XML tags as follows:\n<tool_call>\n{tool_call}\n</tool_call>Also, before making a call to a function take the time to plan the function to take. Make that thinking process between <think>{your thoughts}</think>\n\nI need a random number between 1 and 100.
"""

messages = [
    {"role": "user", "content": prompt}
]

text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)
model_inputs = tokenizer([text], return_tensors="pt").to(merged_model.device)

# conduct text completion
generated_ids = merged_model.generate(
    **model_inputs,
    max_new_tokens=16384
)
output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist() 

content = tokenizer.decode(output_ids, skip_special_tokens=True)

print("content:", content)
