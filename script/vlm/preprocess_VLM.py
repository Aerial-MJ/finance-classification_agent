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

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": "./test1.jpg",
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
print(output_text)
