# -*- coding: utf-8 -*-
"""
Agentic RAG MVP（真实向量库版）
支持：检索 → 查看文件 → 精读 chunk → 引用 source+line
"""

import json
import os
from typing import List, Dict
from dataclasses import dataclass

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

# ==================== 配置 ====================
PERSIST_DIR = "/data/postgraduates/2024/chenjiarui/Model/Agent/script/rag/data/chroma_db"
EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-0.6B"
EMBEDDING_BASE_URL = "https://api.siliconflow.cn/v1"
EMBEDDING_API_KEY = "sk-tgprnspwkhliprfcuobqpfiiwjawxkgaldpfkjtovpfudpmf"

LLM_MODEL = "deepseek-chat"
LLM_BASE_URL = "https://api.deepseek.com"
LLM_API_KEY = "sk-9fc40e8ded4a45f5b9fc61b3330074d3"

# 检查向量库
if not os.path.exists(PERSIST_DIR):
    raise FileNotFoundError(f"向量库不存在: {PERSIST_DIR}")


# ==================== 初始化嵌入 + 向量库 ====================
embeddings = OpenAIEmbeddings(
    base_url=EMBEDDING_BASE_URL,
    model=EMBEDDING_MODEL,
    api_key=EMBEDDING_API_KEY
)

vectorstore = Chroma(persist_directory=PERSIST_DIR, embedding_function=embeddings)
print(f"向量库加载成功！共 {vectorstore._collection.count()} 条记录")


# ==================== 工具实现（真实向量库） ====================

@tool("query_knowledge_base")
def query_knowledge_base(query: str) -> str:
    """在知识库中执行语义检索，返回前4个最相关的 chunk（含 source、line、preview）"""
    docs = vectorstore.similarity_search(query, k=4)
    results = []
    for i, doc in enumerate(docs):
        kf = doc.metadata.get("key_fields", {})
        if isinstance(kf, str):
            try: kf = json.loads(kf)
            except: kf = {}
        key_preview = " | ".join(f"{k}:{v}" for k, v in list(kf.items())[:3] if v)

        results.append({
            "rank": i + 1,
            "source": doc.metadata.get("source", "unknown"),
            "line": doc.metadata.get("line", "?"),
            "label": doc.metadata.get("label", "N/A"),
            "preview": doc.page_content.strip()[:150] + ("..." if len(doc.page_content) > 150 else ""),
            "key_fields": key_preview
        })
    return json.dumps(results, ensure_ascii=False, indent=2)


@tool("get_files_meta")
def get_files_meta(file_sources: List[str]) -> str:
    """根据 source 文件名获取元信息（模拟 getFilesMeta）"""
    if not file_sources:
        return "请提供 source 文件名列表"
    # 统计每个 source 出现的次数（模拟 chunk_count）
    from collections import Counter
    source_count = Counter(doc.metadata.get("source") for doc in vectorstore._collection.get()["documents"])
    results = []
    for src in file_sources:
        count = source_count.get(src, 0)
        results.append({
            "filename": src,
            "chunk_count": count,
            "status": "done"
        })
    return json.dumps(results, ensure_ascii=False, indent=2)


@tool("read_file_chunks")
def read_file_chunks(chunks: List[Dict[str, str]]) -> str:
    """根据 source + line 读取完整 chunk 内容"""
    if not chunks:
        return "请提供 chunk 信息数组（source + line）"
    results = []
    for spec in chunks:
        source = spec.get("source")
        line = spec.get("line")
        if not source or line is None:
            continue
        # 精确查找
        docs = vectorstore._collection.get(where={"source": source, "line": line}, limit=1)
        if docs["documents"]:
            content = docs["documents"][0]
            metadata = docs["metadatas"][0]
            results.append({
                "source": source,
                "line": line,
                "content": content,
                "label": metadata.get("label", "N/A")
            })
    return json.dumps(results, ensure_ascii=False, indent=2)


@tool("list_files")
def list_files(page: int = 0, page_size: int = 10) -> str:
    """列出知识库中所有 source 文件（去重 + 分页）"""
    docs = vectorstore._collection.get()
    sources = sorted(set(docs["metadatas"][i].get("source") for i in range(len(docs["metadatas"]))))
    start = page * page_size
    end = start + page_size
    page_sources = sources[start:end]
    results = []
    for src in page_sources:
        count = sum(1 for m in docs["metadatas"] if m.get("source") == src)
        results.append({
            "filename": src,
            "chunk_count": count,
            "status": "done"
        })
    return json.dumps(results, ensure_ascii=False, indent=2)


# ==================== 创建 Agent ====================
def create_agentic_rag_agent():
    tools = [query_knowledge_base, get_files_meta, read_file_chunks]
    # tools = [query_knowledge_base, get_files_meta, read_file_chunks, list_files]

    # 自定义系统指令（插入到模板开头）
    SYSTEM_PROMPT ="""你是一个 Agentic RAG 助手。请严格遵循以下“先粗后细”策略：

    1. 先用 **query_knowledge_base** 搜索相关内容，获取候选 chunk（source + line）
    2. 必要时用 **get_files_meta** 查看文件规模
    3. 选择 **1~3 个最相关的 chunk**，用 **read_file_chunks** 精读完整内容
    4. 基于 **实际读取的内容** 组织答案
    5. 回答末尾必须写：
       > 引用：
       > - source: xxx.jsonl (第 x 行)
       > - source: yyy.jsonl (第 y 行)
    6. **最多 3 步后停止！**

    重要：
    - 禁止编造信息
    - 若证据不足，请说“知识库中未找到足够信息”
    - 优先选择 label 匹配、key_fields 相关的 chunk
    """
    
    llm=ChatOpenAI(model=LLM_MODEL, api_key=LLM_API_KEY, base_url=LLM_BASE_URL)

    agent = create_react_agent(llm, tools, prompt=SYSTEM_PROMPT)
    return agent


# ==================== 主函数：invoke_rag_model ====================
def invoke_rag_model(query: str) -> str:
    """对外接口：输入问题 → 返回带引用的答案"""
    agent = create_agentic_rag_agent()

    print(f"\n问题: {query}")
    print("="*80)
    print("Agent 思考过程：")
    print("-"*50)

    result = agent.invoke({"messages": [("user", query)]})
    for res in result["messages"]:
        print(res)
        


# ==================== 测试 ====================
if __name__ == "__main__":
    invoke_rag_model("营业执照的注册资本是多少？")