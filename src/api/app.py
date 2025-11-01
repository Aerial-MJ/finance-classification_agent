from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
from Agent.src.chain_define import (
    invoke_VLM_model, invoke_classification_model, invoke_deepseek_model, invoke_model,
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
        result = invoke_VLM_model(request.image_rotate_path)
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
        classification = invoke_model(request.image_rotate_path)
        # ocr_layoutlm(request.image_rotate_path)
        result={
            "image1":"/data/postgraduates/2024/chenjiarui/Model/Agent/src/result/ner_output/ner_annotated.png",
            "classification":classification
        }
        return {"status": "success", "classification_result": result}
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



if __name__ == "__main__":
    os.system("uvicorn app:app --host 0.0.0.0 --port 8000")