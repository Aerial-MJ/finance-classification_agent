"""
Minimal Knowledge Base Tools for LLM Agent
(无 RAG，仅函数注册和模拟数据)
"""

import json
from typing import List, Dict
from dataclasses import dataclass

# ======== 模拟知识库控制器 ========
@dataclass
class FileChunk:
    file_id: int
    chunk_index: int
    content: str

@dataclass
class FileInfo:
    id: int
    filename: str
    chunk_count: int
    status: str = "done"

class MockKnowledgeBaseController:
    """内存版知识库控制器，仅供 Agent 工具调用演示"""

    def __init__(self):
        self.files = [
            FileInfo(1, "intro_to_ai.md", 3),
            FileInfo(2, "deep_learning.md", 4),
            FileInfo(3, "transformer_architecture.md", 3),
        ]

        self.chunks = {
            (1, 0): FileChunk(1, 0, "AI（人工智能）是一种让机器具备人类智能的技术。"),
            (1, 1): FileChunk(1, 1, "人工智能的应用包括自动驾驶、语音识别、图像分析等。"),
            (1, 2): FileChunk(1, 2, "AI 的发展依赖于算力、算法和数据。"),
            (2, 0): FileChunk(2, 0, "深度学习是一种基于多层神经网络的机器学习方法。"),
            (2, 1): FileChunk(2, 1, "常见的深度学习框架包括 TensorFlow、PyTorch。"),
            (3, 0): FileChunk(3, 0, "Transformer 是一种自注意力机制模型，用于处理序列数据。"),
            (3, 1): FileChunk(3, 1, "Transformer 广泛应用于 NLP，如 BERT 和 GPT。"),
        }

    def listFilesPaginated(self, page: int, page_size: int) -> List[Dict]:
        start = page * page_size
        end = start + page_size
        files_slice = self.files[start:end]
        return [f.__dict__ for f in files_slice]

    def getFilesMeta(self, file_ids: List[int]) -> List[Dict]:
        return [f.__dict__ for f in self.files if f.id in file_ids]

    def readFileChunks(self, chunks: List[Dict[str, int]]) -> List[Dict]:
        result = []
        for chunk_spec in chunks:
            key = (chunk_spec.get("fileId"), chunk_spec.get("chunkIndex"))
            if key in self.chunks:
                c = self.chunks[key]
                result.append(c.__dict__)
        return result

    def search(self, query: str) -> List[Dict]:
        """简单关键字匹配搜索"""
        results = []
        for (fid, idx), chunk in self.chunks.items():
            if query.lower() in chunk.content.lower():
                results.append({
                    "file_id": fid,
                    "chunk_index": idx,
                    "preview": chunk.content[:100],
                    "filename": next(f.filename for f in self.files if f.id == fid)
                })
        return results[:5]

# ======== 工具函数 ========
kb_controller = MockKnowledgeBaseController()

def query_knowledge_base(query: str) -> str:
    """Query the mock knowledge base."""
    results = kb_controller.search(query)
    return json.dumps(results, ensure_ascii=False, indent=2)

def get_files_meta(fileIds: List[int]) -> str:
    """Get metadata for files."""
    results = kb_controller.getFilesMeta(fileIds)
    return json.dumps(results, ensure_ascii=False, indent=2)

def read_file_chunks(chunks: List[Dict[str, int]]) -> str:
    """Read file chunks from the knowledge base."""
    results = kb_controller.readFileChunks(chunks)
    return json.dumps(results, ensure_ascii=False, indent=2)

def list_files(page: int = 0, pageSize: int = 10) -> str:
    """List files in the knowledge base."""
    results = kb_controller.listFilesPaginated(page, pageSize)
    return json.dumps(results, ensure_ascii=False, indent=2)

# ======== 注册为 LLM 可调用的工具 ========
tools = [
    {
        "type": "function",
        "function": {
            "name": "query_knowledge_base",
            "description": "Search the knowledge base by query text",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string", "description": "Search query"}},
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_files_meta",
            "description": "Get metadata for specific files",
            "parameters": {
                "type": "object",
                "properties": {"fileIds": {"type": "array", "items": {"type": "integer"}}},
                "required": ["fileIds"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file_chunks",
            "description": "Read content chunks from files",
            "parameters": {
                "type": "object",
                "properties": {
                    "chunks": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "fileId": {"type": "integer"},
                                "chunkIndex": {"type": "integer"},
                            },
                        },
                    }
                },
                "required": ["chunks"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": "List files with pagination",
            "parameters": {
                "type": "object",
                "properties": {
                    "page": {"type": "integer"},
                    "pageSize": {"type": "integer"},
                },
                "required": [],
            },
        },
    },
]

if __name__ == "__main__":
    # 测试一下
    print(list_files())
    print(query_knowledge_base("Transformer"))
