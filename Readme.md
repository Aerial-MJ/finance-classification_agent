
# 文档智能分类系统（Document Intelligence Classification System）

一个基于 **多模态大模型 + OCR + NER + RAG + 分类融合** 的端到端文档理解与分类系统，支持图像输入，自动完成 **图像预处理 → OCR识别 → 布局分析 → 实体抽取 → 文档分类** 全流程。

---

![前端界面](./utils/image/front1.png)


## 项目结构

```text
.
├── data/                   # 示例图片数据
├── script/                 # 各模块实现
│   ├── classification/     # 图像+文本融合分类模型
│   ├── deepseek/           # 最终决策 LLM（DeepSeek）
│   ├── functioncall/       # 函数调用 Agent（未启用）
│   ├── paddle/             # PaddleOCR 相关
│   ├── rag/                # 向量检索增强（RAG）
│   └── vlm/                # 视觉语言模型（Qwen2.5-VL）
├── src/
│   ├── chain_define.py     # 核心流程函数封装（OCR、NER、VLM、分类、RAG）
│   ├── main.py             # 主入口脚本
│   ├── result/             # 输出结果目录（自动生成）
│   ├── test.jpg            # 测试图片
│   └── tools.json          # 函数工具定义（Agent 使用）
├── test_tool/              # 实验性工具脚本
├── utils/                  # Jupyter 实验与工具函数
└── README.md

````

## 功能亮点

* 图像预处理：自动旋转校正（PaddleOCR DocPreprocessor）
* OCR识别：高精度中英文 OCR，支持倾斜文本布局
* NER：LayoutLMv3 识别 HEADER/QUESTION/ANSWER 实体
* VLM 理解：Qwen2.5-VL 提取关键字段、表格、标题、类型
* 分类模型：ResNet + BERT 图像文本融合分类（置信度 >90% 直接输出）
* RAG 增强：Chroma 向量库检索相似模板
* 最终决策：DeepSeek LLM 综合多源信息输出最终分类

## 支持的文档类型

* 业务委托书-处理
* 利润表-处理
* 特种转账借方-处理
* 特种转账贷方-处理
* 营业执照-处理
* 资产负债表-处理
* 身份证反面
* 身份证正面-处理
* 转账支票-处理
* 进账单-处理

也支持未见过的类型，由 VLM + LLM 推理得出。

---

## 环境要求

* Python >= 3.8
* PyTorch >= 1.12
* CUDA >= 11.3 (推荐)

---

## 依赖安装

```bash
pip install -r requirements.txt
```

模型会自动从 HuggingFace 或 HF-Mirror 下载（已配置镜像加速）。

---

## 快速开始

1. **克隆项目**

```bash
git clone https://github.com/Aerial-MJ/finance-classification_agent.git
```

2. **准备测试图片**

将图片放入 `src/` 目录，或修改 `main.py` 中的路径：

```python
parser.add_argument("--image", type=str, default="./test.jpg")
```

3. **运行系统**

```bash
python src/main.py --image ./test.jpg
```

---

## 示例输出

```text
============================================================
图片预处理并识别ocr文字
============================================================
执行分类模型
  1. 转账支票-处理                    0.9421 (94.21%)
  2. 进账单-处理                      0.0312 (3.12%)
  3. 业务委托书-处理                   0.0120 (1.20%)
当前置信度较高，直接分类为 转账支票-处理

若置信度 < 90%，系统将进入 多模型融合决策流程。
```

---

## 输出结果

运行后会在 `src/result/` 生成：

* `preprocess_image.jpg`：旋转校正后图像
* `ocr_output/`：OCR 可视化与 JSON
* `ner_output/`：NER 标注图像（彩色）
* `ner_output/ner_results.json`：实体识别结果
* 终端打印最终分类结果

---

## 核心流程详解（chain_define.py）

```python
image_rotate() 
→ invoke_orc_model() 
→ invoke_classification_model() 
   └─> 置信度 >90% → 直接返回
   └─> 否则 → 多模型融合
→ invoke_ocr_layoutLMv3_model()  # NER
→ invoke_VLM_model()             # 关键字段 + 布局
→ invoke_rag_model()             # 检索模板
→ invoke_deepseek_model()        # 最终决策
```

---

## 模型说明（请提前下载）

* **Qwen2.5-VL**：
* **paddle ocr**：
* **LayoutLMv3**：
* **classification_model**：
* **RAG 向量库**：

如需本地运行，请修改 `chain_define.py` 中的路径。

---

## 四大功能
### 知识库检索
![知识库检索](./utils/image/knowledg.png)

### 布局分析
![布局分析](./utils/image/layout.png)

### 图片识别

![图片识别](./utils/image/recognition.png)
### 图片分类

![图片分类](./utils/image/classification.png)


## 未来可扩展

* 支持 PDF 多页输入
* 增量学习新文档类型

---


## 致谢

* PaddleOCR
* Transformers
* Qwen-VL
* LangChain
* DeepSeek

---

## License

MIT

欢迎提交 Issue 或 PR，一起打造企业级文档智能引擎！

