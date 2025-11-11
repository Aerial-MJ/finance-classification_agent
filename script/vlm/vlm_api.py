import os
import base64
from io import BytesIO
from PIL import Image  # 新增依赖Pillow库
from openai import OpenAI
import json
from Agent.configs.parse import args

# 新增：将任意格式图片转换为内存中的PNG二进制数据
def convert_to_png_bytes(image_path):
    """将任意格式图片转换为内存中的PNG格式二进制数据"""
    try:
        with Image.open(image_path) as img:
            # 处理透明通道（确保兼容性）
            if img.mode in ('RGBA', 'LA') or (img.mode == 'P' and 'transparency' in img.info):
                img = img.convert('RGBA')
            else:
                img = img.convert('RGB')  # 非透明图转RGB
            
            # 存入内存缓冲区（不落地保存）
            png_buffer = BytesIO()
            img.save(png_buffer, format='PNG')  # 强制转为PNG格式
            return png_buffer.getvalue()  # 返回PNG二进制数据
    except Exception as e:
        print(f"图片格式转换失败：{str(e)}")
        return None

# 修改：基于PNG二进制数据进行base64编码
def encode_image_to_base64(png_bytes):
    """将PNG二进制数据编码为base64字符串"""
    return base64.b64encode(png_bytes).decode("utf-8")


client = OpenAI(
    api_key= args.vlm_api_key,  # 替换为你的API Key
    base_url= args.vlm_base_url,
)


prompt = """
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
"""

# 输入图片路径（支持任意格式：jpg、png、bmp、tiff等）
image_path = "test1.jpg"  # 可替换为test.png、test.bmp、test.tiff等

# 核心流程：先转PNG，再编码，最后调用模型
png_bytes = convert_to_png_bytes(image_path)
if not png_bytes:
    print("无法处理图片，请检查路径或格式")
else:
    base64_image = encode_image_to_base64(png_bytes)
    
    # 调用文生图模型
    completion_image2text = client.chat.completions.create(
        model="qwen3-vl-plus",
        messages=[
            {
                "role": "system",
                "content": [{"type":"text","text": "你是一位非常擅长理解图片的助手"}]
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{base64_image}"},  # 始终以PNG格式传入
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ],
    )
    
    # 提取结果
    text = completion_image2text.choices[0].message.content
    
    print(text)