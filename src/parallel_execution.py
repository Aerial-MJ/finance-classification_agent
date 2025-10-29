from concurrent.futures import ThreadPoolExecutor, as_completed , ProcessPoolExecutor
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from Agent.src.chain_define import invoke_VLM_model,invoke_classification_model,invoke_deepseek_model,invoke_orc_model,invoke_ocr_layoutLMv3_model,invoke_rag_model,image_rotate
import argparse
import re
import json
import traceback

def thread1_pipeline(rotate_image, ocr_text):
    print("="*60)
    print("执行分类模型")
    classification_result = invoke_classification_model(rotate_image, ocr_text)

    for idx, res in enumerate(classification_result):
        key, value = list(res.keys())[0], list(res.values())[0]
        print(f"  {idx+1}. {key:<30} {float(value)/100:.4f} ({value}%)")

    cl = classification_result[0]
    class_, score = list(cl.keys())[0], list(cl.values())[0]

    if float(score) > 90:
        print("当前置信度较高，直接分类为", class_)
        return {"high_confidence": True, "class": class_}

    print("当前置信度较低，建议进一步调用模型,可能不属于该分类")

    classification_text = "分类模型预测结果（按置信度排序）:\n"
    for idx, res in enumerate(classification_result):
        key, value = list(res.items())[0]
        classification_text += f"{idx+1}. {key} ({value}%)\n"

    print("="*60)
    print("执行ocr——ner模型")
    ocr_json = invoke_ocr_layoutLMv3_model(rotate_image)

    return {
        "high_confidence": False,
        "classification_result": classification_result,
        "classification_text": classification_text,
        "ocr_json": ocr_json,
    }

def thread2_pipeline(rotate_image):
    print("="*60)
    print("执行vlm模型")
    VLM_json = invoke_VLM_model(rotate_image)
    print("="*60)
    print("执行rag模型")
    rag_text = invoke_rag_model(str(VLM_json))
    return {"VLM_json": VLM_json, "rag_text": rag_text}

# ======================
# 主线程调度
# ======================

with ThreadPoolExecutor(max_workers=2) as executor:
    os.makedirs("./result", exist_ok=True)
    os.makedirs("./result/ocr_output", exist_ok=True)
    os.makedirs("./result/ner_output", exist_ok=True)
    
    parser = argparse.ArgumentParser(description="Single-sample inference for classification model")
    
    parser.add_argument(
        "--image",
        type=str,
        help="Path to input image",
        default="./test.jpg"
    )

    args = parser.parse_args()

    print("="*60)
    print("图片预处理并识别ocr文字")

    rotate_image = image_rotate(args.image)
    ocr_text = invoke_orc_model(rotate_image)
    print("="*60)
    print("开始并行执行")
    future1 = executor.submit(thread1_pipeline, rotate_image, ocr_text)
    future2 = executor.submit(thread2_pipeline, rotate_image)

    results = {}
    with ProcessPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(thread1_pipeline, rotate_image, ocr_text),
            executor.submit(thread2_pipeline, rotate_image)
        ]

        for future in as_completed(futures):
            try:
                res = future.result()
                results.update(res)
            except Exception as e:
                print("❌ 子进程出错:", e)
                traceback.print_exc()

    print("="*60)
    print("全部进程执行完毕")

# ======================
# 如果置信度高，直接退出
# ======================
if results.get("high_confidence"):
    print("最终类别为：", results["class"])
else:
    print("="*60)
    print("执行最终的判断模型")

    final_prompt = (
        "你是一个文档分类模型，需要根据以下信息判断该文档属于哪一类：\n\n"
        f"1. VLM 模型提取的关键信息和布局特征：\n{results['VLM_json']}\n\n"
        f"2. OCR 与 NER 提取的文本信息和命名实体信息：\n{results['ocr_json']}\n\n"
        "请着重注意HEADER标签，是很清晰的分类依据。\n\n"
        f"3. 检索到的相关背景知识：\n{results['rag_text']}\n\n"
        f"4. 分类模型的预测结果：\n{results['classification_text']}\n\n"
        "文档可能属于以下分类之一：业务委托书-处理, 利润表-处理, 特种转账借方-处理, "
        "特种转账贷方-处理, 营业执照-处理, 资产负债表--处理, 身份证反面, 身份证正面--处理, "
        "转账支票-处理, 进账单-处理，也有可能属于其他未列出的分类。\n"
        "请输出文档的最终分类名称，只输出分类，不要其他解释。"
    )

    result = invoke_deepseek_model(final_prompt)
    print("最终的类别是：")
    print(result)
