# -*- coding: utf-8 -*-
"""
快速测试已构建的 RAG 向量库
路径: /data/postgraduates/2024/chenjiarui/Model/Agent/script/rag/data/chroma_db
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
PERSIST_DIR = "/data/postgraduates/2024/chenjiarui/Model/Agent/script/rag/data/chroma_db_final"

EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-0.6B"
BASE_URL = "https://api.siliconflow.cn/v1"
API_KEY = "sk-tgprnspwkhliprfcuobqpfiiwjawxkgaldpfkjtovpfudpmf"  # 你的 key

# 检查路径
if not os.path.exists(PERSIST_DIR):
    raise FileNotFoundError(f"向量库不存在: {PERSIST_DIR}")

# ==================== 加载向量库 ====================
print("正在加载向量库...")
embeddings = OpenAIEmbeddings(
    base_url=BASE_URL,
    model=EMBEDDING_MODEL,
    api_key=API_KEY
)

vectorstore = Chroma(persist_directory=PERSIST_DIR, embedding_function=embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

query = "江南农村商业银行的贷款合同有哪些关键信息？"
retrieved_docs = retriever.get_relevant_documents(query)

print(f"向量库加载成功！共 {vectorstore._collection.count()} 条记录\n")


# ==================== 测试查询 ====================
queries = []

print("="*60)
print("开始 RAG 检索测试")
print("="*60)

for i, query in enumerate(queries, 1):
    print(f"\n查询 {i}: {query}")
    print("-" * 50)
    
    # 关键：执行检索，返回 Document 列表
    docs = vectorstore.similarity_search(query, k=4)
    
    if not docs:
        print("未检索到任何结果")
        continue

    retrieve_content=""

    # 遍历检索结果
    for j, doc in enumerate(docs, 1):
        # 1. 内容预览
        content = doc.page_content.strip()
        preview = content[:300] + ("..." if len(content) > 300 else "")
        preview = preview.replace("\n", " ").replace("  ", " ")

        # 2. 关键元数据
        source = doc.metadata.get('source', 'unknown')
        line = doc.metadata.get('line', '?')
        label = doc.metadata.get('label', 'N/A')
        file_type = doc.metadata.get('file_type', 'N/A')
        has_table = doc.metadata.get('layout_has_table', '未知')
        
        # 3. key_fields 处理
        kf = doc.metadata.get('key_fields', {})
        if isinstance(kf, str):
            try:
                kf = json.loads(kf)
            except:
                kf = {}
        key_preview = " | ".join(f"{k}:{v}" for k, v in list(kf.items()) if v)

        
        # 4. 打印
        print(f"[{j}] {preview}")
        print(f"    → 来源: {source} (第 {line} 行)")
        print(f"    → 标签: {label}")
        if key_preview:
            print(f"    → 关键: {key_preview}")
        retrieve_content += (
            f"[{j}] {preview}\n"
            f"    → 来源: {source} (第 {line} 行)\n"
            f"    → 标签: {label}\n"
            f"    → 关键: {key_preview}\n"
        )

    print("-" * 80)


print("\n✅ RAG 测试完成！")