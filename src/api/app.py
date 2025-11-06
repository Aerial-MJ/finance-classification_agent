import re
import shutil
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
from Agent.src.chain_define import (
    invoke_VLM_Local_model, invoke_VLM_model, invoke_classification_model, invoke_deepseek_model, invoke_model,
    invoke_orc_model, invoke_ocr_layoutLMv3_model, invoke_rag_model, image_rotate
)
UPLOAD_DIR = "./uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import os
import uuid
import base64
from io import BytesIO
from pathlib import Path
import json

app = FastAPI()

# 允许所有来源（所有端口、所有域名）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 允许所有的来源
    allow_credentials=True,
    allow_methods=["*"],  # 允许所有的 HTTP 方法
    allow_headers=["*"],  # 允许所有的请求头
)

# Create a Pydantic model to handle incoming requests (for text, file, etc.)
class ImageRequest(BaseModel):
    image_path: str

class OCRRequest(BaseModel):
    image_rotate_path: str

class ClassificationRequest(BaseModel):
    image_rotate_path: str
    ocr_text: str

class RagRequest(BaseModel):
    rag_text: str

class DeepseekRequest(BaseModel):
    text: str


current_directory = os.path.dirname(os.path.abspath(__file__))
KNOWLEDGE_BASE_PATH = "/data/postgraduates/2024/chenjiarui/Model/Agent/script/rag/data/knowledge_base.jsonl"
PREPROCESS_IMAGE = "/data/postgraduates/2024/chenjiarui/Model/Agent/src/result/preprocess_image.jpg"

@app.post("/upload-file")
async def upload_file(file: UploadFile = File(...)):
    # 生成唯一文件名
    file_ext = file.filename.split(".")[-1]
    print(file)
    print(file_ext)
    print(file.filename)
    new_filename = f"{uuid.uuid4()}.{file_ext}"
    file_path = os.path.join(UPLOAD_DIR, new_filename)
    # 保存文件
    with open(file_path, "wb") as f:
        content = await file.read()
        f.write(content)

    upload_path = os.path.join(current_directory, "uploads", new_filename)
    image_path = image_rotate(upload_path)

    # 返回相对路径（前端拼接 httpURL 使用）
    return {"file_path": image_path }






# Image Rotation API Endpoint
@app.post("/rotate-image")
async def rotate_image(request: ImageRequest):
    try:
        print(request.image_path)
        rotated_image_path = image_rotate(request.image_path)
        return {"status": "success", "rotated_image_path": rotated_image_path}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))






# OCR Model API Endpoint
@app.post("/ocr")
async def ocr(request: OCRRequest):
    try:
        ocr_text = invoke_orc_model(request.image_rotate_path)
        return {"status": "success", "ocr_text": ocr_text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))






# VLM Model API Endpoint
@app.post("/vlm")
async def vlm(request: OCRRequest):
    try:
        result = invoke_VLM_Local_model(request.image_rotate_path)
        return {"status": "success", "result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))





# LayoutLMv3 Model API Endpoint
@app.post("/ocr_layoutlm")
async def ocr_layoutlm(request: OCRRequest):
    try:
        invoke_ocr_layoutLMv3_model(request.image_rotate_path)
        result={
            "image1":"/data/postgraduates/2024/chenjiarui/Model/Agent/src/result/ner_output/ner_annotated.png",
            "image2":"/data/postgraduates/2024/chenjiarui/Model/Agent/src/result/ocr_output/preprocess_image_ocr_res_img.jpg",
        }
        return {"status": "success", "result": result}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))




# Classification Model API Endpoint
@app.post("/classification")
async def classification(request: OCRRequest):
    try:
        rotate_image = request.image_rotate_path  # 原始上传路径
        print("="*60)
        print("图片预处理并识别ocr文字")


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
            invoke_ocr_layoutLMv3_model(rotate_image)
            return class_
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
        VLM_json = invoke_VLM_Local_model(rotate_image)
        # VLM_json = "{'file_type': '健康确认表', 'key_fields': {'姓名': '李秀', '性别': '女', '出生日期': '1982.4.13', '国籍': '中国', '联系电话': '15195851025', '护照号码': '568924', '有效签证': '有', '现居住地址': '成都市东湖国际东光-琉璃路299号', '代理人姓名': '', '代理人证件及号码': '', '填表日期': '', '申请人签名': '', '经办人签名': '', '审核人签署': ''}, 'layout_features': {'has_table': True, 'has_title': True, 'title': '健在确认表（存根）', 'table_structure': '多列多行表格，包含姓名、性别、出生日期、国籍、联系电话、提交证件情况、现居住地址、代理人情况、填表日期、申请人签名、经办人签名、审核人签署等字段'}, 'content_summary': '该文件是一份健康确认表，用于记录个人的基本信息、联系方式、证件情况以及居住地址等。表格中包含了姓名、性别、出生日期、国籍、联系电话、提交证件情况、现居住地址、代理人情况、填表日期、申请人签名、经办人签名、审核人签署等字段。'}"

        print("="*60)
        print("执行rag模型")
        rag_text = invoke_rag_model(VLM_json["file_type"])

        print("="*60)
        print("执行最终的判断模型")    
        image_dir = "/data/postgraduates/2024/chenjiarui/Model/Agent/script/rag/data/图片示例"

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
        classification=invoke_deepseek_model(final_prompt)

        x = 0
        valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}
        target_dir="/data/postgraduates/2024/chenjiarui/Model/Agent/script/rag/data/图片示例/"+classification
        if not os.path.exists(target_dir):
            x = 1
        else:
            for file in os.listdir(target_dir):
                file_lower = file.lower()
                if file_lower.startswith("image_") and any(file_lower.endswith(ext) for ext in valid_extensions):
                    x += 1
            x += 1


        data = {
            "image_path": target_dir + f"/image_{x}.jpg",  # 原始路径
            "label": classification,   # 最终分类结果
            "VLM_text": {
                "file_type": classification,
                "key_fields": VLM_json.get("key_fields", {}),
                "content_summary": VLM_json.get("content_summary", f"分类结果: {classification}")
            }
        }

        # 4. 构造返回结果（含图片 + 分类 + 完整 data）
        result = {
            "image1": "/data/postgraduates/2024/chenjiarui/Model/Agent/src/result/ner_output/ner_annotated.png",
            "classification": classification,
            "knowledge_base_data": data
        }

        return {
            "status": "success",
            "classification_result": result
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))





# RAG Model API Endpoint
@app.post("/rag")
async def rag(request: RagRequest):
    print(request.rag_text)
    try:
        result = invoke_rag_model(request.rag_text , "api")
        return {"status": "success", "result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))




# Deepseek Model API Endpoint
@app.post("/deepseek")
async def deepseek(request: DeepseekRequest):
    try:
        result = invoke_deepseek_model(request.text)
        return {"status": "success", "result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))





@app.post("/image")
async def get_image(request: ImageRequest):
    """
    返回图片的 Base64 编码
    前端用: <img :src="data:image/jpeg;base64,xxx">
    """
    print(request)
    file_path = request.image_path
    print(file_path)
    
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    
    # 读取图片并转 Base64
    with open(file_path, "rb") as f:
        image_data = f.read()
        image_base64 = base64.b64encode(image_data).decode('utf-8')
    
    # 根据文件扩展名设置 MIME 类型
    ext = file_path.lower().split('.')[-1]
    mime_types = {
        'jpg': 'image/jpeg',
        'jpeg': 'image/jpeg', 
        'png': 'image/png',
        'gif': 'image/gif'
    }
    mime_type = mime_types.get(ext, 'image/jpeg')
    
    return {
        "image_base64": f"data:{mime_type};base64,{image_base64}"
    }




@app.post("/save-to-knowledge-base")
async def save_to_knowledge_base(data: dict):
    """
    1. 把 preprocess_image.jpg 复制到 data["image_path"]
    2. 追加写入 knowledge_base.jsonl
    """
    try:
        # 1. 验证 image_path
        image_path = data.get("image_path")
        if not image_path:
            raise HTTPException(status_code=400, detail="image_path 缺失")

        target_path = Path(image_path)
        
        # 确保目标目录存在
        target_path.parent.mkdir(parents=True, exist_ok=True)

        # 2. 复制 preprocess_image.jpg 到目标路径
        if Path(PREPROCESS_IMAGE).exists():
            shutil.copy2(PREPROCESS_IMAGE, target_path)
            print(f"预处理图片已复制: {PREPROCESS_IMAGE} → {target_path}")
        else:
            print(f"警告: 预处理图片不存在: {PREPROCESS_IMAGE}")

        # 3. 追加写入 JSONL
        with open(KNOWLEDGE_BASE_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(data, ensure_ascii=False) + "\n")

        return {"status": "success", "message": "保存成功，已复制预处理图片"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    os.system("uvicorn app:app --host 0.0.0.0 --port 8000")