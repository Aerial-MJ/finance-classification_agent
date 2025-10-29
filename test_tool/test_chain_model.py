import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import json
import re

# 模型和 tokenizer 路径（假设训练后保存的位置）
model_path = "/data/postgraduates/2024/chenjiarui/Model/Agent/script/functioncall/fineturn_functioncall/qwen_functioncall_full_finetune/checkpoint-2000"
device = "cuda" if torch.cuda.is_available() else "cpu"

# 加载 tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

# 加载模型（全参数微调后的模型，使用 bfloat16 或 float16 以节省内存）
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="auto",
    trust_remote_code=True
)
model.eval()

# 创建文本生成 pipeline
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=1024,
    temperature=0.1,
    do_sample=True,
    return_full_text=False
)

# 定义工具（基于你的对话示例，使用 mock 实现）
def get_stock_price(company: str) -> str:
    # Mock 实现，返回模拟股票价格
    mock_prices = {"Apple": "$150.75", "Microsoft": "$210.22"}
    return mock_prices.get(company, "Unknown company")

def get_movie_details(title: str) -> str:
    # Mock 实现，返回模拟电影详情
    mock_details = {"Inception": "Director: Christopher Nolan, Year: 2010, Genre: Sci-Fi"}
    return mock_details.get(title, "Unknown movie")


def add_numbers(a: int, b: int) -> int:
    return a + b

tools = [
    {
        "type": "function",
        "function": {
            "name": "add_numbers",
            "description": "Add two numbers",
            "parameters": {
                "type": "object",
                "properties": {
                    "a": {"type": "number", "description": "First number"},
                    "b": {"type": "number", "description": "Second number"}
                },
                "required": ["a", "b"]
            }
        }
    }
]

# 系统提示（基于你的训练数据）
system_prompt = """
You are a function calling AI model. You are provided with function signatures within <tools></tools> XML tags. You may call one or more functions to assist with the user query. Don't make assumptions about what values to plug into functions. Here are the available tools: <tools> {tools} </tools> Use the following pydantic model json schema for each tool call you will make: {'title': 'FunctionCall', 'type': 'object', 'properties': {'arguments': {'title': 'Arguments', 'type': 'object'}, 'name': {'title': 'Name', 'type': 'string'}}, 'required': ['arguments', 'name']} For each function call return a json object with function name and arguments within <tool_call></tool_call> XML tags as follows:
<tool_call>
{tool_call}
</tool_call> Also, before making a call to a function take the time to plan the function to take. Make that thinking process between <think>{your thoughts}</think>
"""
import ast
# 解析模型输出中的 <think> 和 <tool_call>
def parse_output(output: str):
    think_match = re.search(r"<think>(.*?)</think>", output, re.DOTALL)
    think = think_match.group(1).strip() if think_match else None
    
    tool_call_match = re.search(r"<tool_call>(.*?)</tool_call>", output, re.DOTALL)
    if tool_call_match:
        try:
            tool_call_str = tool_call_match.group(1).strip()
            tool_call_json = ast.literal_eval(tool_call_str)
            
            return {
                "think": think,
                "tool_name": tool_call_json.get("name"),
                "tool_args": tool_call_json.get("arguments", {})
            }
        except json.JSONDecodeError:
            return {"think": think, "error": "Invalid JSON in tool_call"}
    return {"think": think, "response": output}

# 执行工具
def execute_tool(tool_name: str, tool_args: dict):
    if tool_name == "get_stock_price":
        company = tool_args.get("company")
        if company:
            return {"stock_price": get_stock_price(company)}
        else:
            return {"error": "Missing 'company' argument"}
    elif tool_name == "get_movie_details":
        title = tool_args.get("title")
        if title:
            return {"details": get_movie_details(title)}
        else:
            return {"error": "Missing 'title' argument"}
    elif tool_name == "add_numbers":
        a = tool_args.get("a")
        b = tool_args.get("b")
        return {"result": add_numbers(a, b)} if a is not None and b is not None else {"error": "Missing a or b"}
    else:
        return {"error": "Unknown tool"}

# Agent 主函数：处理多轮对话
def agent_chat(user_input: str, history: list = None):
    print(history)
    print("------------------------------------")
    
    if len(history)==0 :
        history = []

        # 构建工具字符串
        tools_str = json.dumps(tools)
    
        # 构建系统提示
        full_system = system_prompt.replace("{tools}", tools_str)
    
        # 构建消息列表：系统 + 历史 + 当前用户输入
        messages = [{"role": "user", "content": full_system + user_input}]
        history.append(messages[0])
        print(history)
        print("------------------------------------")
    
    else:
        messages = history + [{"role": "user", "content": user_input}]
    
    # 使用 apply_chat_template 格式化提示
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    # 生成输出
    output = pipe(prompt)[0]["generated_text"]
    
    print(output)

    parsed = parse_output(output)
    print(parsed)

    if "tool_name" in parsed:
        # 有工具调用
        print(f"Agent Thinking: {parsed['think']}")
        tool_result = execute_tool(parsed["tool_name"], parsed["tool_args"])
        print(f"Tool Result: {tool_result}")
        
        # 构建工具响应消息
        tool_response_content = f"<tool_response>\n{json.dumps(tool_result)}\n</tool_response>"
        history.append({"role": "assistant", "content": output})
        history.append({"role": "tool", "content": tool_response_content})
        
        # 继续生成最终回复
        messages = history
        final_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        final_output = pipe(final_prompt)[0]["generated_text"]
        
        # 解析最终输出
        final_parsed = parse_output(final_output)
        response = final_parsed.get("response", final_output)
        
        history.append({"role": "assistant", "content": final_output})
    print(f"Agent Response: {response}")
    return response, history

if __name__ == "__main__":
    history = []
    
    # 第一轮：查询 Apple 股票
    user_input = "What is 1 + 1?"
    response, history = agent_chat(user_input, history)   
    print(history)



    