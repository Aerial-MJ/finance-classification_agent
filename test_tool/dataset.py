import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import (
    LayoutLMv3ForTokenClassification,
    LayoutLMv3ImageProcessor,
    XLMRobertaTokenizerFast,
    get_linear_schedule_with_warmup,
)
from datasets import Dataset , load_from_disk
from PIL import Image
from tqdm import tqdm
import json
import os

# ===== 1. 配置参数（不变）=====
class Config:
    model_name = "../layoutlmv3-base-chinese"
    labels = ["O", "B-HEADER", "I-HEADER", "B-QUESTION", "I-QUESTION", "B-ANSWER", "I-ANSWER"]
    num_labels = len(labels)
    batch_size = 2
    num_epochs = 10
    learning_rate = 5e-5
    max_length = 512
    image_size = (224, 224)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = "./layoutlmv3-chinese-trained"

# ===== 2. 数据预处理（🔥 单个样本版）=====
def prepare_examples(examples):
    tokenizer = XLMRobertaTokenizerFast.from_pretrained(Config.model_name)
    image_processor = LayoutLMv3ImageProcessor.from_pretrained(Config.model_name, apply_ocr=False)
    
    img = examples["image"]
    words = examples["words"]
    boxes = examples["bboxes"]
    labels_list = examples["ner_tags"]

    if isinstance(img, str):
        img = Image.open(img)
    img = img.convert("RGB")

    pixel_values = image_processor(images=[img], return_tensors="pt")["pixel_values"].squeeze(0)

    w, h = img.size
    normalized_boxes = [
        [int(1000 * x / w), int(1000 * y / h), int(1000 * x2 / w), int(1000 * y2 / h)]
        for x, y, x2, y2 in boxes
    ]

    encoding = tokenizer(
        words,
        truncation=True,
        padding="max_length",
        max_length=Config.max_length,
        return_tensors="pt",
        is_split_into_words=True
    )

    input_ids = encoding["input_ids"].squeeze(0)
    attention_mask = encoding["attention_mask"].squeeze(0)

    bbox_tensor = torch.zeros(Config.max_length, 4, dtype=torch.int64)
    labels_tensor = torch.full((Config.max_length,), -100, dtype=torch.long)

    word_ids = encoding.word_ids(batch_index=0)
    for idx, word_idx in enumerate(word_ids):
        if word_idx is not None and word_idx < len(normalized_boxes):
            bbox_tensor[idx] = torch.tensor(normalized_boxes[word_idx])
            labels_tensor[idx] = labels_list[word_idx]

    return {
        "pixel_values": pixel_values,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "bbox": bbox_tensor,
        "labels": labels_tensor
    }

# ===== 3. 加载本地数据（🔥 batched=False）=====
def load_and_prepare_data(json_path="zh.train.json", img_dir="zh"):
    print(f"📥 读取 {json_path} ...")
    with open(json_path, "r", encoding="utf-8") as f:
        file_data = json.load(f)

    all_ids, all_words, all_bboxes, all_labels, all_images = [], [], [], [], []
    label_map = {"header": (1, 2), "question": (3, 4), "answer": (5, 6), "other": (0, 0)}

    for doc in file_data["documents"]:
        img_path = os.path.join(img_dir, f"{doc['id']}.jpg")
        image = Image.open(img_path).convert("RGB")

        words, bboxes, ner_tags = [], [], []
        for element in doc["document"]:
            b_tag, i_tag = label_map.get(element["label"], (0, 0))
            for idx, word in enumerate(element["words"]):
                words.append(word["text"])
                x0, y0, x1, y1 = word["box"]
                bboxes.append([x0, y0, x1, y1])
                ner_tags.append(b_tag if idx == 0 else i_tag)

        all_ids.append(doc["id"])
        all_words.append(words)
        all_bboxes.append(bboxes)
        all_labels.append(ner_tags)
        all_images.append(image)

    dataset = Dataset.from_dict({
        "id": all_ids,
        "image": all_images,
        "words": all_words,
        "bboxes": all_bboxes,
        "ner_tags": all_labels
    })

    split = dataset.train_test_split(test_size=0.2, seed=42)
    
    train_dataset = dataset.map(
        prepare_examples, 
        batched=False,  # 🔥 关键修复
        remove_columns=["id", "image", "words", "bboxes", "ner_tags"]
    )
    
   
    train_dataset.save_to_disk("./processed_train_dataset")
  

load_and_prepare_data()