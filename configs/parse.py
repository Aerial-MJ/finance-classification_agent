import argparse
import os
import sys

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

def get_argparse(prog_name="DocumentIntelligenceSystem"):

    parser = argparse.ArgumentParser(prog=prog_name)

    # ------------------ Paths ------------------
    parser.add_argument("--data_dir", type=str,  default=os.path.join(BASE_DIR, "Agent/data/图片示例"), help="官方数据目录")
    parser.add_argument("--rag_data_dir", type=str,  default=os.path.join(BASE_DIR, "Agent/script/rag/data/图片示例"), help="rag知识库目录")
    parser.add_argument("--class_data_dir", type=str,  default=os.path.join(BASE_DIR, "Agent/script/classification/data_raw"), help="classification模型数据目录")
    parser.add_argument("--script_dir", type=str, default=os.path.join(BASE_DIR,"Agent/script"), help="模型脚本目录")
    parser.add_argument("--src_dir", type=str, default=os.path.join(BASE_DIR,"Agent/src"), help="源代码目录")
    parser.add_argument("--test_dir", type=str, default=os.path.join(BASE_DIR,"Agent/test"), help="零样本测试目录")
    parser.add_argument("--knowledge_base_dir", type=str, default=os.path.join(BASE_DIR, "Agent/script/rag/data/knowledge_base.jsonl"), help="RAG知识库目录")
    parser.add_argument("--knowledge_base_class",type=str,default=os.path.join(BASE_DIR, "Agent/script/rag/data/knowledge_class.jsonl"),help="RAG 知识分类 JSONL 文件路径")
    parser.add_argument("--persist_dir",type=str,default=os.path.join(BASE_DIR, "Agent/script/rag/data/chroma_db_final"),help="Chroma 向量数据库持久化路径")
    parser.add_argument("--local_persist_dir",type=str,default=os.path.join(BASE_DIR, "Agent/script/rag/data/chroma_db"),help="Chroma 向量数据库持久化路径")


    # ------------------ Model Selection ------------------
    parser.add_argument("--ocr_model", type=str, default=os.path.join(BASE_DIR, "Paddle/PaddleOCR"), help="OCR模型")
    parser.add_argument("--vlm_model", type=str, default=os.path.join(BASE_DIR, "Qwen/Qwen2.5-VL-7B-Instruct"), help="Qwen本地多模态模型")
    parser.add_argument("--layoutLMv3_train_model", type=str, default=os.path.join(BASE_DIR, "LayoutLMv3/layoutlmv3-chinese/layoutlmv3-chinese-trained/best_model"), help="布局分析训练模型")
    parser.add_argument("--layoutLMv3_base_model", type=str, default=os.path.join(BASE_DIR, "LayoutLMv3/layoutlmv3-base-chinese"), help="布局分析初始模型")
    parser.add_argument("--bert_base_chinese", type=str, default=os.path.join(BASE_DIR, "Agent/script/classification/bert-base-chinese"), help="bert 预训练模型")
    parser.add_argument("--classification_model", type=str, default=os.path.join(BASE_DIR, "Agent/script/classification/kfold_checkpoints/fold_3_best.pt"), help="bert 预训练模型")   
    parser.add_argument("--qwen3_1_7b_model",type=str,default=os.path.join(BASE_DIR, "Qwen/Qwen3-1.7B"),help="Qwen3-1.7B 模型本地路径")
    parser.add_argument("--functioncall_model_path", type=str, default=os.path.join(BASE_DIR, "Agent/script/fineturn_functioncall/qwen_functioncall_full_finetune/checkpoint-2000"), help="Fine-tuned Qwen model checkpoint")
    parser.add_argument("--distill_teacher_model",type=str,default=os.path.join(BASE_DIR, "Qwen/DeepSeek-R1-Distill-Qwen-1.5B"),help="DeepSeek-R1-Distill 模型本地路径")
    parser.add_argument("--distill_student_model",type=str,default=os.path.join(BASE_DIR, "Qwen/Qwen2.5-1.5B"),help="Qwen2.5-1.5B 模型本地路径")
 

    # ------------------ api key ------------------
    parser.add_argument("--rag_base_url",type=str,default="https://api.siliconflow.cn/v1",help="API 基础访问地址")
    parser.add_argument("--rag_api_key",type=str,default=os.getenv("SILICONFLOW_API_KEY", "sk-tgprnspwkhliprfcuobqpfiiwjawxkgaldpfkjtovpfudpmf"),help="SiliconFlow API 密钥（可从环境变量 SILICONFLOW_API_KEY 读取）")
    parser.add_argument("--embedding_model",type=str,default="Qwen/Qwen3-Embedding-0.6B",help="嵌入向量模型（Embedding Model）")
    parser.add_argument("--llm_model", type=str, default="deepseek-chat", help="LLM 模型名称")
    parser.add_argument("--llm_base_url", type=str, default="https://api.deepseek.com", help="LLM 服务的基础 URL")
    parser.add_argument("--llm_api_key", type=str, default=os.getenv("LLM_API_KEY", "sk-9fc40e8ded4a45f5b9fc61b3330074d3"), help="LLM API 密钥（可从环境变量 LLM_API_KEY 读取）")
    parser.add_argument("--vlm_api_key", type=str, default=os.getenv("API_KEY", "sk-b95eb1a0a35f44efa7b49d2bca9d4c1f"), help="API 密钥（可从环境变量 API_KEY 读取）")
    parser.add_argument("--vlm_base_url", type=str, default="https://dashscope.aliyuncs.com/compatible-mode/v1", help="API 服务的基础 URL")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--reload", action="store_true")
    
    
    # ------------------ image  ------------------
    parser.add_argument("--image1", type=str, default=os.path.join(BASE_DIR, "Agent/src/result/ner_output/ner_annotated.png"), help="路径到ner_annotated.png")
    parser.add_argument("--image2", type=str, default=os.path.join(BASE_DIR, "Agent/src/result/ocr_output/preprocess_image_ocr_res_img.jpg"), help="路径到preprocess_image_ocr_res_img.jpg")
    parser.add_argument("--preprocess_image", type=str, default=os.path.join(BASE_DIR, "Agent/src/result/preprocess_image.jpg"), help="路径到预处理图片")


    return parser

parser = get_argparse()
args = parser.parse_args()