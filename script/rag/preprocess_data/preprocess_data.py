import os
import shutil
import random
import json
from Agent.configs.parse import args

# 源路径和目标路径
source_root = args.data_dir
target_root = args.rag_data_dir
jsonl_path = args.knowledge_base_dir

# 创建目标根目录
os.makedirs(target_root, exist_ok=True)

# 用于存jsonl
knowledge_base = []

# 遍历每个类别文件夹
for class_name in os.listdir(source_root):
    class_path = os.path.join(source_root, class_name)
    if not os.path.isdir(class_path):
        continue

    # 创建类别目标文件夹
    target_class_path = os.path.join(target_root, class_name)
    os.makedirs(target_class_path, exist_ok=True)

    # 列出所有图片
    images = [f for f in os.listdir(class_path) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    images = sorted(images)  # 保证顺序可复现
    # 随机挑选 10 张
    selected_images = images[:10] if len(images) <= 10 else random.sample(images, 10)

    # 拷贝并重命名
    for idx, img_name in enumerate(selected_images, start=1):
        src_path = os.path.join(class_path, img_name)
        dst_name = f"image_{idx}.jpg"
        dst_path = os.path.join(target_class_path, dst_name)
        shutil.copy(src_path, dst_path)

        # 添加到知识库 jsonl
        record = {
            "image_path": dst_path,
            "label": class_name,
            "VLM_text": ""
        }
        knowledge_base.append(record)

# 写入 jsonl 文件
with open(jsonl_path, "w", encoding="utf-8") as f:
    for record in knowledge_base:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

print(f"完成！知识库已生成：{jsonl_path}")

