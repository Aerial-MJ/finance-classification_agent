import ast
import json
import os
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info

# default: Load the model on the available device(s)
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "/data/postgraduates/2024/chenjiarui/Model/Qwen/Qwen2.5-VL-7B-Instruct", torch_dtype="auto", device_map="auto" 
)

# We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
# model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
#     "Qwen/Qwen2.5-VL-7B-Instruct",
#     torch_dtype=torch.bfloat16,
#     attn_implementation="flash_attention_2",
#     device_map="auto",
# )

# default processer
processor = AutoProcessor.from_pretrained("/data/postgraduates/2024/chenjiarui/Model/Qwen/Qwen2.5-VL-7B-Instruct")

# The default range for the number of visual tokens per image in the model is 4-16384.
# You can set min_pixels and max_pixels according to your needs, such as a token range of 256-1280, to balance performance and cost.
# min_pixels = 256*28*28
# max_pixels = 1280*28*28
# processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct", min_pixels=min_pixels, max_pixels=max_pixels)
output_file = "/data/postgraduates/2024/chenjiarui/Model/Agent/script/rag/data/knowledge_base.jsonl"


def save_knowledge_base(image):
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image,
                },
                {"type": "text", 
                "text": """
        请分析下图，为了方便判断它属于以下哪种类型的文件，
        我需要你提取该文件中的关键信息，包括文件的关键字段（如：姓名、日期、金额、编号等）、文本内容、相关布局特征（如：表格、段落、标题等）。
        请按以下格式返回输出：
        {
            "file_type": "文件类型（需要你自己判断）,
            "key_fields": {
                "field_1": "字段值",
                "field_2": "字段值",
                ...
            },
            "layout_features": {
                "has_table": true/false,
                "has_title": true/false,
                "title": "标题文本",
                "table_structure": "表格结构描述（如：列名、行数等）"
            },
            "content_summary": "文件内容简要概述"
        }
        """}
            ],
        }
    ]

    # Preparation for inference
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")

    # Inference: Generation of the output
    generated_ids = model.generate(**inputs, max_new_tokens=512)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )

    def extract_json_from_text(text: str):
        """从文本中提取第一个 {...} 并解析成 JSON"""
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or start >= end:
            # 找不到合法 JSON
            return None
        json_str = text[start:end+1]
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            return None
        
    result = extract_json_from_text(output_text[0])

    print(result)
    return (str(result))


image_dir = "/data/postgraduates/2024/chenjiarui/Model/Agent/script/rag/data/图片示例"

# 读取图片文件
def process_images(image_dir):
    # 遍历文件夹中的所有子文件夹和图片文件
    image_paths = []
    for root, dirs, files in os.walk(image_dir):
        for file in files:
            if file == "preprocessed.png": # 假设图片是 PNG 格式
                image_paths.append(os.path.join(root, file))
    return image_paths

image_path=process_images(image_dir)

import json
import os

image_list = []

output_file = "/data/postgraduates/2024/chenjiarui/Model/Agent/script/rag/data/knowledge_base.jsonl"

# 1. 读取原始 knowledge_base.jsonl
with open(output_file, "r", encoding="utf-8") as f:
    knowledge_list = [json.loads(line) for line in f]
    for list_image in knowledge_list:
        image_list.append(list_image["image_path"][:-4]+"/output/preprocessed.png")

print(image_list)


# 2. 遍历图片，调用 VLM 并更新对应 entry
for preprocessed_path in image_list:
    # 获取原始 jpg 图片路径
    dir_path = os.path.dirname(preprocessed_path)  # .../image_1/output
    parent_dir = os.path.dirname(dir_path)  # .../image_1
    image_name = os.path.basename(parent_dir)  # image_1
    category_dir = os.path.dirname(parent_dir)  # .../营业执照-处理
    image_i_path = os.path.join(category_dir, image_name + ".jpg")  # .../营业执照-处理/image_1.jpg

    # 调用 VLM
    vlm_text = save_knowledge_base(preprocessed_path)

    # 更新 knowledge_list
    for entry in knowledge_list:
        if entry["image_path"] == image_i_path:
            entry["VLM_text"] = vlm_text
            break

# 3. 写回 jsonl
with open(output_file, "w", encoding="utf-8") as f:
    for entry in knowledge_list:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")



with open(output_file, "r", encoding="utf-8") as f:
    knowledge_list = [json.loads(line) for line in f]
    for entry in knowledge_list:
        VLM_text = ast.literal_eval(entry["VLM_text"])
        label=entry["label"]
        VLM_text["file_type"]=label
        entry["VLM_text"] = VLM_text

with open(output_file, "w", encoding="utf-8") as f:
    for entry in knowledge_list:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n") 
        
    