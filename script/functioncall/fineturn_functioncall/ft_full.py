from enum import Enum
import torch

from datasets import load_dataset
from trl import SFTConfig, SFTTrainer
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    TrainingArguments, set_seed,
    DataCollatorForLanguageModeling
)

# ====== 基础配置 ======
seed = 42
set_seed(seed)

model_name = "/data/postgraduates/2024/chenjiarui/Model/Qwen/Qwen3-1.7B"
dataset_name = "./hermes-function-calling-thinking-V1"
output_dir = "./qwen_functioncall_full_finetune"
per_device_train_batch_size = 1  # 可根据显存调整
gradient_accumulation_steps = 4
learning_rate = 1e-4  # 全参数微调建议比LoRA更小一点

# ====== Tokenizer 设置 ======
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.chat_template = "{{ bos_token }}{% if messages[0]['role'] == 'system' %}{{ raise_exception('System role not supported') }}{% endif %}{% for message in messages %}{{ '<|im_start|>' + message['role'] + '\n' + message['content'] | trim + '<|im_end|>\n' }}{% endfor %}{% if add_generation_prompt %}{{'<|im_start|>assistant\n'}}{% endif %}"

# ====== 数据加载 ======
dataset = load_dataset(dataset_name)
dataset = dataset.rename_column("conversations", "messages")

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
    if first_message["role"] == "system":
        system_message_content = first_message["content"]
        messages[1]["content"] = (
            system_message_content
            + "Also, before making a call to a function take the time to plan the function to take. "
              "Make that thinking process between <think>{your thoughts}</think>\n\n"
            + messages[1]["content"]
        )
        messages.pop(0)
    return {"text": tokenizer.apply_chat_template(messages, tokenize=False)}

dataset = dataset.map(preprocess, remove_columns="messages")
dataset_train = dataset["train"]

# ====== 特殊Token定义 ======
class ChatmlSpecialTokens(str, Enum):
    tools = "<tools>"
    eotools = "</tools>"
    think = "<think>"
    eothink = "</think>"
    tool_call = "<tool_call>"
    eotool_call = "</tool_call>"
    tool_response = "<tool_response>"
    eotool_response = "</tool_response>"
    pad_token = "<|endoftext|>"
    eos_token = "<|im_end|>"

    @classmethod
    def list(cls):
        return [c.value for c in cls]

tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    # pad_token=ChatmlSpecialTokens.pad_token.value,
    # additional_special_tokens=ChatmlSpecialTokens.list()
)

tokenizer.chat_template = "{{ bos_token }}{% if messages[0]['role'] == 'system' %}{{ raise_exception('System role not supported') }}{% endif %}{% for message in messages %}{{ '<|im_start|>' + message['role'] + '\n' + message['content'] | trim + '<|im_end|>\n' }}{% endfor %}{% if add_generation_prompt %}{{'<|im_start|>assistant\n'}}{% endif %}"


if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# ====== 模型加载（全参数微调）======
# 🚫 不使用 LoRA，直接加载模型进行全量微调
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
    device_map="auto",
    trust_remote_code=True
)

# ====== 训练参数 ======
training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=per_device_train_batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    learning_rate=learning_rate,
    num_train_epochs=3,
    logging_steps=10,
    save_steps=500,
    save_strategy="steps",
    bf16=True,
    optim="adamw_torch",
    report_to="none",
    gradient_checkpointing=True,   # ✅ 建议开启以节省显存
    warmup_ratio=0.03,
    lr_scheduler_type="cosine",
)

# ====== 数据整理器 ======
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# ====== Trainer 配置 ======
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset_train,
    tokenizer=tokenizer,
    args=training_args,
    dataset_text_field="text",
    max_seq_length=4096,
    data_collator=data_collator,
)


batch = trainer.get_train_dataloader().dataset[0]
print("""batch["input_ids"]=""",batch["input_ids"])
print("--------------------------------------")
print("""batch["input_ids"]=""",tokenizer.decode(batch["input_ids"]))
print(batch["labels"])

# ====== 开始训练 ======
print("🚀 开始全参数微调...")
print(dataset_train[0]["text"][:4096])
trainer.train()

# ====== 保存模型 ======
trainer.save_model("./results_full_finetune")
tokenizer.save_pretrained("./results_full_finetune")
print(f"✅ 全参数微调完成！模型保存在: ./results_full_finetune")
