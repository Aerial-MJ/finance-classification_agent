import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from Agent.src.chain_define import invoke_VLM_model,invoke_classification_model,invoke_deepseek_model,invoke_orc_model,invoke_ocr_layoutLMv3_model,invoke_rag_model,image_rotate
import argparse
from Agent.configs.parse import args
from pathlib import Path

class Config:

    system_prompt = """
    You are a function calling AI model. You are provided with function signatures within <tools></tools> XML tags. You may call one or more functions to assist with the user query. Don't make assumptions about what values to plug into functions. Here are the available tools: <tools> {tools} </tools> Use the following pydantic model json schema for each tool call you will make: {'title': 'FunctionCall', 'type': 'object', 'properties': {'arguments': {'title': 'Arguments', 'type': 'object'}, 'name': {'title': 'Name', 'type': 'string'}}, 'required': ['arguments', 'name']} For each function call return a json object with function name and arguments within <tool_call></tool_call> XML tags as follows:
    <tool_call>
    {tool_call}
    </tool_call> Also, before making a call to a function take the time to plan the function to take. Make that thinking process between <think>{your thoughts}</think>
    """
    model_path = args.qwen3_1_7b_model
    device = "cuda" if torch.cuda.is_available() else "cpu"


def agent_chat(user_input: str = None, history: list = None):
    tokenizer = AutoTokenizer.from_pretrained(Config.model_path, trust_remote_code=True)

    ft_model = AutoModelForCausalLM.from_pretrained(
        Config.model_path,
        device_map="auto",
        trust_remote_code=True
    )
    ft_model.eval()

    pipe = pipeline(
        "text-generation",
        model=ft_model,
        tokenizer=tokenizer,
        max_new_tokens=1024,
        temperature=0,
        do_sample=True,
        return_full_text=False
    )

    messages = history
    
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    output = pipe(prompt)[0]["generated_text"]

    # print(output)
    return output




def main():
    os.makedirs("./result", exist_ok=True)
    os.makedirs("./result/ocr_output", exist_ok=True)
    os.makedirs("./result/ner_output", exist_ok=True)

    # with open("./tools.json", "r", encoding="utf-8") as f:
    #     tools_data = json.load(f)
    #     full_system = Config.system_prompt.replace("{tools}", json.dumps(tools_data["tools"]))
    
    parser = argparse.ArgumentParser(description="Single-sample inference for classification model")
    

    parser.add_argument(
        "--image",
        type=str,
        help="Path to input image",
        default="./test.jpg"
    )

    args = parser.parse_args()

    # print(full_system)
    # history = []
    # user_input ="""what is OCR and named entity recognition of the image?"""
    # history.append({"role": "user", "content": full_system + user_input})

    # agent_chat(history=history)
    
    print("="*60)
    print("图片预处理并识别ocr文字")

    rotate_image = image_rotate(args.image)
    ocr_text = invoke_orc_model(rotate_image)
    print("="*60)
    print("执行分类模型")

    classification_result = invoke_classification_model(rotate_image,ocr_text)
    # print(classification_result)
    for idx , res in enumerate(classification_result) :
        key , value = list(res.keys())[0],list(res.values())[0]
        print(f"  {idx+1}. {key:<30} {float(value)/100:.4f} ({value}%)")

    cl = classification_result[0]
    class_ , score =list(cl.keys())[0] , list(cl.values())[0]

    if(float(score)>90):
        print("当前置信度较高，直接分类为", class_)
        return
    else:
        print("当前置信度较低，建议进一步调用模型,可能不属于该分类")

    classification_text = "分类模型预测结果（按置信度排序）:\n"
    for idx, res in enumerate(classification_result):
        key, value = list(res.items())[0]
        classification_text += f"{idx+1}. {key} ({value}%)\n"
    print("="*60)
    print("执行ocr——ner模型")
    ocr_json = invoke_ocr_layoutLMv3_model(rotate_image)


    print("="*60)
    print("执行vlm模型")
    VLM_json = invoke_VLM_model(rotate_image)

    print("="*60)
    print("执行rag模型")
    rag_text = invoke_rag_model(str(VLM_json))

    print("="*60)
    print("执行最终的判断模型")   

    image_dir = args.rag_data_dir
    categories = [p.name for p in Path(image_dir).iterdir() if p.is_dir()]
    categories_str = ",".join(categories)


    final_prompt = (
        "你是一个文档分类模型，需要根据以下信息判断该文档属于哪一类：\n\n"
        f"1. VLM 模型提取的关键信息和布局特征：\n{VLM_json}\n\n"
        f"2. OCR 与 NER 提取的文本信息和命名实体信息：\n{ocr_json}\n\n"
        "请着重注意HEADER标签，是很清晰的分类依据\n"
        f"3. 检索到的相关背景知识（可以作为参考）：\n{rag_text}\n\n"
        f"4. 先前分类模型的预测结果（因为置信度较低，所以没有直接作为判断依据）：\n{classification_text}\n\n"
        f"文档可能属于以下分类之一：{categories_str}。\n"
        "也可能属于其他未列出的分类。\n"
        "请输出文档的最终分类名称，只输出分类，不要其他解释。"
    )

    result=invoke_deepseek_model(final_prompt)
    print("最终的类别是：")
    print(result)

    
if __name__ == "__main__":
    main()