from enum import Enum
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
from Agent.configs.parse import args

seed = 42
set_seed(seed)

model_name = args.qwen3_1_7b_model
dataset_name = "./hermes-function-calling-thinking-V1"
tokenizer = AutoTokenizer.from_pretrained(model_name)
output_dir = "./qwen_functioncall"
per_device_train_batch_size = 1  # 根据显存调整
gradient_accumulation_steps = 4
learning_rate = 1e-4


# 备注：这里的 model 
tokenizer.chat_template = "{{ bos_token }}{% if messages[0]['role'] == 'system' %}{{ raise_exception('System role not supported') }}{% endif %}{% for message in messages %}{{ '<|im_start|>' + message['role'] + '\n' + message['content'] | trim + '<|im_end|>\n' }}{% endfor %}{% if add_generation_prompt %}{{'<|im_start|>assistant\n'}}{% endif %}"


dataset = load_dataset(dataset_name)
dataset = dataset.rename_column("conversations", "messages")

# 改成和 qwen 对应 assistant / user 的格式
def convert_model_to_assistant(sample):
    messages = sample["messages"]
    for message in messages:
        if message["role"] == "model":
            message["role"] = "assistant"
        if message["role"] == "human":
            message["role"] = "user"
    return sample

dataset = dataset.map(convert_model_to_assistant)


def preprocess(sample):
    messages = sample["messages"]
    first_message = messages[0]

    # Instead of adding a system message, we merge the content into the first user message
    if first_message["role"] == "system":
        system_message_content = first_message["content"]
        # Merge system content with the first user message
        messages[1]["content"] = system_message_content + "Also, before making a call to a function take the time to plan the function to take. Make that thinking process between <think>{your thoughts}</think>\n\n" + messages[1]["content"]
        # Remove the system message from the conversation
        messages.pop(0)

    return {"text": tokenizer.apply_chat_template(messages, tokenize=False)}


dataset = dataset.map(preprocess, remove_columns="messages")

# dataset = dataset["train"].train_test_split(0.1)
# dataset = dataset["train"]

class ChatmlSpecialTokens(str, Enum):
    tools = "<tools>"
    eotools = "</tools>"
    think = "<think>"
    eothink = "</think>"
    tool_call="<tool_call>"
    eotool_call="</tool_call>"
    tool_response="<tool_response>"
    eotool_response="</tool_response>"
    pad_token = "<|endoftext|>"
    eos_token = "<|im_end|>"
    @classmethod
    def list(cls):
        return [c.value for c in cls]

tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        pad_token=ChatmlSpecialTokens.pad_token.value,
        additional_special_tokens=ChatmlSpecialTokens.list()
    )

tokenizer.chat_template = "{{ bos_token }}{% if messages[0]['role'] == 'system' %}{{ raise_exception('System role not supported') }}{% endif %}{% for message in messages %}{{ '<|im_start|>' + message['role'] + '\n' + message['content'] | trim + '<|im_end|>\n' }}{% endfor %}{% if add_generation_prompt %}{{'<|im_start|>assistant\n'}}{% endif %}"

dataset_train = dataset["train"]

# dataset_val = dataset["test"]

# 1. 添加pad_token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# 2. LoRA配置
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
    r=16,
    lora_alpha=32,
    lora_dropout=0.1,
    # target_modules=["q_proj","v_proj","k_proj"]
    target_modules=['k_proj', 'gate_proj', 'v_proj', 'up_proj', 'q_proj', 'o_proj', 'down_proj']
)


# 3. model配置
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)

model = get_peft_model(model, lora_config)


# 4. 训练参数
training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=per_device_train_batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    learning_rate=learning_rate,
    optim="paged_adamw_32bit",
    num_train_epochs=3,
    logging_steps=10,
    save_steps=100,
    save_strategy="steps",
    # eval_steps=500,
    # eval_strategy="steps", 
    bf16=True,               
    report_to="none"      
)

# 5. 数据整理器
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)


# 6. 训练器
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset_train,
    tokenizer=tokenizer,
    # eval_dataset=dataset_val,
    args=training_args,
    dataset_text_field="text",
    max_seq_length=4096
)

# 7. 开始训练
print("🚀 开始微调...")

print(dataset_train[0]["text"][:4096])
trainer.train()

# 8. 保存模型
trainer.save_model("./results")
tokenizer.save_pretrained("./results")
print(f"✅ 微调完成！模型保存在: ./results")