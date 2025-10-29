# -*- coding: utf-8 -*-
"""
构建 RAG 向量库
支持多个 JSONL 文件（每行一个 dict）
"""

import json
import os
from typing import List
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

# ==================== 配置 ====================
# 1. 你的两个 JSONL 文件（路径请自行确认是否重复）
JSONL_PATHS = [
    "/data/postgraduates/2024/chenjiarui/Model/Agent/script/rag/data/knowledge_base.jsonl",
    "/data/postgraduates/2024/chenjiarui/Model/Agent/script/rag/data/knowledge_class.jsonl", 
]

# 2. 向量库持久化目录
PERSIST_DIR = "../data/chroma_db"

# 3. 嵌入模型（SiliconFlow）
EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-0.6B"
BASE_URL = "https://api.siliconflow.cn/v1"

# ==================== 1. 加载 JSONL ====================
def load_jsonl_documents(file_paths: List[str]) -> List[Document]:
    docs = []
    seen = set()  # 去重 (source + line)

    fp1=file_paths[0]
    fp2=file_paths[1]
    with open(fp2, "r", encoding="utf-8") as f:
        knowledge_list = [json.loads(line) for line in f]
        knowledge_dic={}
        for li in knowledge_list:
            knowledge_dic[list(li.keys())[0]] = list(li.values())[0]
    print(knowledge_dic)

    with open(fp1, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)

                # ---- 提取 VLM_text（已经是 dict）----
                vlm = data.get("VLM_text", {})
                source = data.get("image_path", {})
                if isinstance(vlm, str):
                    try:
                        vlm = json.loads(vlm)
                    except:
                        vlm = {}

                # ---- 文本内容 ----
                content = vlm.get("content_summary", "").strip()
                if not content:
                    # 若 summary 为空，用 key_fields 拼接
                    kf = vlm.get("key_fields", {})
                    content = " | ".join(
                        f"{k}: {v}" for k, v in kf.items() if v and v != ""
                    )

                if not content:
                    continue  # 空文档直接跳过

                # ---- 元数据 ----
                metadata = {
                    "source": source,
                    "line": line_no,
                    "label": data.get("label", ""),
                    "key_fields": json.dumps(vlm.get("key_fields", {}), ensure_ascii=False),
                    "label_introduction":"("+data.get("label", "") +")是"+ knowledge_dic[data.get("label", "")]
                }

                print(metadata.get("key_fields"))
                print("----------------------------")

                doc = Document(page_content=content, metadata=metadata)
                docs.append(doc)

            except Exception as e:
                print(f"[{fp1}:{line_no}] 解析错误: {e}")
            
    print(docs[0])
    print(f"共加载 {len(docs)} 条有效文档")
    return docs


# ==================== 主流程 ====================
def main():
    print("开始加载 JSONL...")
    documents = load_jsonl_documents(JSONL_PATHS)
    if not documents:
        raise ValueError("没有加载到任何文档！")

    # ---- 文本切分 ----
    print("文本切分...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        length_function=len,
        add_start_index=True,
    )
    splits = splitter.split_documents(documents)
    print(f"切分后得到 {len(splits)} 个块")

    # ---- 嵌入模型 ----
    print("初始化嵌入模型...")
    embeddings = OpenAIEmbeddings(
        base_url=BASE_URL,
        model=EMBEDDING_MODEL,
        api_key="sk-tgprnspwkhliprfcuobqpfiiwjawxkgaldpfkjtovpfudpmf"
    )

    # ---- 写入/增量更新 Chroma ----
    print("写入 Chroma 向量库...")
    if os.path.exists(PERSIST_DIR):
        print(f"检测到已有库，增量追加 → {PERSIST_DIR}")
        vectorstore = Chroma(persist_directory=PERSIST_DIR, embedding_function=embeddings)
        vectorstore.add_documents(splits)
    else:
        vectorstore = Chroma.from_documents(
            documents=splits,
            embedding=embeddings,
            persist_directory=PERSIST_DIR,
        )

    print(f"向量库构建完成！共 {len(splits)} 条记录 → {PERSIST_DIR}")


if __name__ == "__main__":
    main()