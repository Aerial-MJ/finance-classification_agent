import json
import os
from typing import List, Dict, Any
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
import uuid
from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import InMemoryStore  # 新增
import shutil
import os
import time
from Agent.configs.parse import args

# ==================== 配置 ====================
JSONL_BASE = args.knowledge_base_dir
JSONL_CLASS = args.knowledge_base_class
PERSIST_DIR = args.persist_dir

EMBEDDING_MODEL = args.embedding_model
BASE_URL = args.rag_base_url
API_KEY = args.rag_api_key

CHUNK_SIZE = 600
CHUNK_OVERLAP = 120
TOP_K = 5

class VectorStoreManager:
    def __init__(self, persist_directory: str = PERSIST_DIR):
        self.DIR=PERSIST_DIR+f"/{time.time()}"
        self.embeddings = OpenAIEmbeddings(
            model=EMBEDDING_MODEL,
            base_url=BASE_URL,
            api_key=API_KEY,
        )
        self.vector_store = Chroma(
            persist_directory=self.DIR,
            embedding_function=self.embeddings,
            collection_name="doc_rag",
        )

        # 大块切分器（原逻辑）
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", "\n", "。", "；", "，", " | ", " "],
        )

        # 小块切分器（用于精确检索）
        self.child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=20,   # 更小，适合你的字段长度
            chunk_overlap=5,
            separators=["%", "\n\n", "\n", " | " , "。" ,  "，", " "]
        )

        # ParentDocumentRetriever + InMemoryStore
        self.docstore = InMemoryStore()  # 存储大块
        self.retriever = ParentDocumentRetriever(
            vectorstore=self.vector_store,
            docstore=self.docstore,
            child_splitter=self.child_splitter,
            # parent_splitter=self.text_splitter,  # 可选：大块也切
        )

        print("向量库初始化完成（已启用 ParentDocumentRetriever）")

    # ------------------- 加载 label 解释 -------------------
    def _load_label_intro(self) -> Dict[str, str]:
        intro = {}
        if not os.path.exists(JSONL_CLASS):
            print(f"警告: 未找到 {JSONL_CLASS}，将跳过 label 解释")
            return intro
        with open(JSONL_CLASS, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line: continue
                try:
                    obj = json.loads(line)
                    k, v = list(obj.items())[0]
                    intro[k] = v
                except:
                    continue
        print(f"成功加载 {len(intro)} 条 label 解释")
        return intro

    # ------------------- 加载主文档 -------------------
    def _load_documents(self) -> List[Document]:
        docs: List[Document] = []
        seen = set()
        label_intro = self._load_label_intro()

        if not os.path.exists(JSONL_BASE):
            raise FileNotFoundError(f"未找到主文件: {JSONL_BASE}")

        with open(JSONL_BASE, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                line = line.strip()
                if not line: continue
                try:
                    data = json.loads(line)
                    vlm = data.get("VLM_text", {})
                    if isinstance(vlm, str):
                        vlm = json.loads(vlm)

                    label = data.get("label", "")

                    # 提取关键字段
                    key_fields = vlm.get("key_fields", {})
                    summary = vlm.get("content_summary", "").strip()

                    summary += f"\n % 类别 : {label}  %"

                    # 构建完整内容：摘要 + 关键字段
                    field_text = " | ".join(
                        f"{k}: {v}" for k, v in key_fields.items() if v and str(v).strip()
                    )
                    content = f"{summary}"
                    if field_text:
                        content += f"\n关键字段：{field_text}"

                    if not content.strip():
                        continue

                    # 元数据
                    label = data.get("label", "")
                    source = data.get("image_path", "").split("data/")[-1]
                    metadata = {
                        "file_id": str(uuid.uuid4()),
                        "source": source,
                        "label": label,
                        "label_introduction": label_intro.get(label, '未知文档类型'),
                        "key_fields_json": json.dumps(key_fields, ensure_ascii=False),
                        "line": line_no,
                    }

                    # 去重
                    key = (source, line_no)
                    if key in seen: continue
                    seen.add(key)

                    # 关键：大块用原始 content
                    docs.append(Document(page_content=content, metadata=metadata))

                except Exception as e:
                    print(f"[第 {line_no} 行] 解析失败: {e}")

        print(f"共加载 {len(docs)} 条有效文档")
        return docs

    # ------------------- 添加文档（增量） -------------------
    def add_documents(self) -> int:
        raw_docs = self._load_documents()
        if not raw_docs:
            raise ValueError("没有加载到任何文档")

        # 使用 ParentDocumentRetriever 自动切小块 + 存大块
        self.retriever.add_documents(raw_docs)
        print(f"成功写入向量库（小块用于检索，大块用于返回）")
        return len(raw_docs)  # 返回大块数量

    # ------------------- 检索 -------------------
    def search(self, query: str, top_k: int = 3, min_similarity: float = 0.85) -> List[Dict[str, Any]]:

        print(f"\n执行检索: {query}")

        # 1. 用 retriever 直接获取大块（parent documents）
        parent_docs = self.retriever.invoke(query, limit=top_k)  # 多取一些
        print(len(parent_docs))

        if not parent_docs:
            print("retriever 未返回任何大块")
            return []

        # 2. 用 vector_store 计算小块相似度，聚合到 parent
        child_results = self.vector_store.similarity_search_with_score(query, k=top_k)
        parent_score_map = {}

        for child_doc, score in child_results:
            parent_id = child_doc.metadata.get("file_id")  # 必须有 parent_id！
            print(parent_id)
            if not parent_id:
                continue
            if parent_id not in parent_score_map:
                parent_score_map[parent_id] = {"scores": [], "count": 0}
            parent_score_map[parent_id]["scores"].append(score)
            parent_score_map[parent_id]["count"] += 1

        # 3. 合并：大块 + 平均相似度
        formatted = []
        for parent_doc in parent_docs:
            parent_id = parent_doc.metadata.get("file_id")  # 或者 "parent_id"，看你设置
            if not parent_id:
                continue

            # 计算平均相似度
            if parent_id in parent_score_map:
                scores = parent_score_map[parent_id]["scores"]
                avg_distance = sum(scores) / len(scores)
                similarity = round(1 - avg_distance, 4)
            else:
                similarity = 0.0  # 没命中小块

            # 过滤低相似度
            if similarity < min_similarity:
                continue

            formatted.append({
                "content": parent_doc.page_content,        # 完整大块
                "metadata": parent_doc.metadata,
                "similarity": similarity,
                "child_hits": parent_score_map.get(parent_id, {}).get("count", 0),
            })

        # 4. 按相似度排序 + 限制数量
        formatted = sorted(formatted, key=lambda x: x["similarity"], reverse=True)[:top_k]

        print(f"返回 {len(formatted)} 条高相关完整凭证（similarity >= {min_similarity}）")
        for item in formatted:
            source = item["metadata"].get("source", "unknown")
            print(f"  [相似度: {item['similarity']}] {source[:60]}")

        return formatted


# ==================== 运行入口 ====================
def main():
    print("开始构建 RAG 向量库...")
    manager = VectorStoreManager()

    # 构建/更新向量库
    manager.add_documents()

    # 示例检索
    print("\n" + "="*60)
    print("示例检索（效果展示）")
    print("="*60)

    test_queries = [
        "业务委托书-处理"
    ]

    
    for q in test_queries:
        print(f"\n> 查询: {q}")
        results = manager.search(q, top_k=3)
        for i, r in enumerate(results, 1):
            print(f"\n  [{i}] 完整凭证")
            print("=" * 80)
            
            # 1. 完整 content（不截断）
            print("  内容：")
            print(f"    {r['content']}")
            print()
            
            # 2. 完整 metadata（格式化 JSON）
            print("  元数据：")
            import json
            metadata_str = json.dumps(r['metadata'], ensure_ascii=False, indent=4)
            print(f"    {metadata_str}")
            
            # 3. 来源摘要
            source_file = os.path.basename(r['metadata'].get('source', 'unknown'))
            label = r['metadata'].get('label', '未知')
            print(f"\n  来源: {source_file} | 类型: {label}")
            print("-" * 80)


if __name__ == "__main__":
    main()