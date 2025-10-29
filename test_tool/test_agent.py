import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_huggingface import HuggingFacePipeline, ChatHuggingFace
from langchain_core.tools import tool
from langchain.agents import create_agent
# 步骤 1: 加载 Qwen3 模型
model_name = "/data/postgraduates/2024/chenjiarui/Model/Qwen/Qwen3-4B-Instruct-2507"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)

# 步骤 2: 创建 Pipeline
qwen_pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=512,
    temperature=0.1,
    do_sample=True
)

# 步骤 3: 包装为 LangChain LLM
llm_pipeline = HuggingFacePipeline(pipeline=qwen_pipe)
llm = ChatHuggingFace(llm=llm_pipeline)





# 步骤 4: 定义工具
@tool
def add_numbers(a: int, b: int) -> int:
    """求解两个数相加"""
    return a + b

@tool
def write_file(path: str, content: str) -> str:
    """Write content to a file at the given path. Returns the path to the file."""
    with open(path, "w") as f:
        f.write(content)
    return path

tools = [add_numbers,write_file]

# 步骤 5: 创建 Agent
agent = create_agent(model=llm, tools=tools)

# 步骤 6: 运行 Agent
test_prompt = "将下面的内容分别保存到两个文件a.txt和b.txt中。1.Qwen大模型是国内的出色模型 2.LangChain是一个开源的AI开发框架"

output = agent.invoke({"messages": [{"role": "user", "content": test_prompt}]})  # 修复输入格式

# 步骤 7: 输出结果
for msg in output["messages"]:
    msg.pretty_print()