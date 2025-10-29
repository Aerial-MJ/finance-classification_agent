import numpy as np
import torch
from transformers import (
    LayoutLMv3ForTokenClassification, 
    LayoutLMv3ImageProcessor, 
    LayoutLMv3Tokenizer,  # 🔥 改对了！
    AutoTokenizer,
    XLMRobertaTokenizerFast
)
from PIL import Image, ImageDraw, ImageFont
from paddleocr import PaddleOCR
import json
import os
from tqdm import tqdm
from collections import Counter
import cv2
from paddleocr import DocPreprocessor

# ===== 1. 配置 =====/data/postgraduates/2024/chenjiarui/Model/
class Config:
    model_path = "/data/postgraduates/2024/chenjiarui/Model/LayoutLMv3/layoutlmv3-chinese/layoutlmv3-chinese-trained/best_f1_0.8733"  # 你的训练模型
    base_model = "/data/postgraduates/2024/chenjiarui/Model/LayoutLMv3/layoutlmv3-base-chinese"  #
    image_path = "./image_6.jpg"       
    output_img = "./output/ner_annotated.png" 
    output_img1 = "./output/ner_annotated_color.png" 
    output_json = "./output/ner_results.json" 
    processored_img="./output/preprocessed.png" 
    label_list = ["O", "B-HEADER", "I-HEADER", "B-QUESTION", "I-QUESTION", "B-ANSWER", "I-ANSWER"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===== 2. 🔥 修复版加载模型 =====
def load_model():
    # 🔥 先从基础模型加载tokenizer（确保有vocab）
    tokenizer = XLMRobertaTokenizerFast.from_pretrained(Config.base_model)
    image_processor = LayoutLMv3ImageProcessor.from_pretrained(Config.base_model, apply_ocr=False)
    
    # 🔥 再加载你的NER模型
    model = LayoutLMv3ForTokenClassification.from_pretrained(Config.model_path).to(Config.device)
    model.eval()
    return tokenizer, image_processor, model


def rotate_Image(image_path):

    doc_preprocessor = DocPreprocessor()

    # 调用 predict() 返回 generator
    results = doc_preprocessor.predict(
        input=image_path,            
        use_doc_orientation_classify=True,   
        use_doc_unwarping=True                
    )

    for res in results:
        angle=res.json["res"]["angle"]
        print(angle)

    img = Image.open(image_path)
    rotated = img.rotate(angle)

    rotated.save(Config.processored_img)

# ===== 3. 🔥 完美PaddleOCR=====
def run_ocr(image_path):
    ocr = PaddleOCR(
        use_doc_orientation_classify=True,
        use_doc_unwarping=False,
        use_textline_orientation=True
    )
    
    result = ocr.predict(input=image_path, return_word_box=True)
    # print(result)
    for res in result:
        # res.print()
        res.save_to_img("output1")
        res.save_to_json("output1")
    
    words, bboxes = [], []

    for page_idx, page_res in enumerate(result):
        print(f"\n🔍 页面 {page_idx+1}: OCRResult 对象")

        # 保存到临时 JSON 并重新加载（兼容 paddleocr 的 save_to_json 逻辑）
        json_path = f"temp_page_{page_idx}.json"
        page_res.save_to_json(json_path)
        with open(json_path, "r", encoding="utf-8") as f:
            ocr_data = json.load(f)
        os.remove(json_path)

        # 确认主内容在 ocr_data["res"] 中
        if "res" in ocr_data:
            ocr_data = ocr_data["res"]

        # 🔥 如果有 token 级结果（word-level），优先使用它
        text_word_boxes = ocr_data.get("text_word_boxes", [])
        text_words = ocr_data.get("text_word", [])

        if text_word_boxes and text_words:
            print(f"✅ 检测到 {len(text_words)} 个 token 级文本块")

            for line_idx, (word_list, box_array) in enumerate(zip(text_words, text_word_boxes)):
                for token, box in zip(word_list, box_array):
                    # box 是 [x1, y1, x2, y2] 或 4x2 点阵
                    if np.array(box).shape == (4, 2):  # 四点坐标
                        xs, ys = box[:, 0], box[:, 1]
                        xmin, ymin, xmax, ymax = xs.min(), ys.min(), xs.max(), ys.max()
                    elif np.array(box).shape == (4,):  # 直接矩形框
                        xmin, ymin, xmax, ymax = box
                    else:
                        continue

                    words.append(token)
                    bboxes.append([int(xmin), int(ymin), int(xmax), int(ymax)])
        else:
            # ⚠️ 退化到行级识别结果（如果 token 级无）
            rec_texts = ocr_data.get("rec_texts", [])
            rec_boxes = ocr_data.get("rec_boxes", [])
            print(f"⚠️ 使用行级识别结果 {len(rec_texts)} 条")

            for text, box in zip(rec_texts, rec_boxes):
                if len(box) == 4:
                    xmin, ymin, xmax, ymax = box
                else:
                    xs, ys = [p[0] for p in box], [p[1] for p in box]
                    xmin, ymin, xmax, ymax = min(xs), min(ys), max(xs), max(ys)
                words.append(text)
                bboxes.append([int(xmin), int(ymin), int(xmax), int(ymax)])

        print(f"✅ OCR识别: {len(words)} 个文本块")
        print(f"📝 前3个文本: {words[:3]}")
        print(f"📝 前3个方框: {bboxes[:3]}")

    return words, bboxes ,text_word_boxes


# ===== 4. 🔥 完美NER预测=====
def predict_ner(tokenizer, image_processor, model, img, words, bboxes , text_word_boxes):
    if not words:
        print("⚠️  无文本，跳过NER")
        return []
    
    print(f"🎯 开始NER预测...")
    pixel_values = image_processor(images=[img], return_tensors="pt")["pixel_values"].to(Config.device)
    w, h = img.size
    
    # 归一化bbox
    normalized_boxes = [
        [int(1000 * x / w), int(1000 * y / h), int(1000 * x2 / w), int(1000 * y2 / h)]
        for x, y, x2, y2 in bboxes
    ]
    
    # Tokenize（去掉boxes参数！）
    encoding = tokenizer(
        words, 
        is_split_into_words=True, 
        return_tensors="pt",
        truncation=True, 
        padding="max_length", 
        max_length=512
    )
    
    input_ids = encoding["input_ids"].to(Config.device)
    attention_mask = encoding["attention_mask"].to(Config.device)
    
    # 手动bbox tensor
    word_ids = encoding.word_ids()
    bbox_tensor = torch.zeros(512, 4, dtype=torch.long)

    for i, word_idx in enumerate(word_ids):
        if word_idx is not None and word_idx < len(normalized_boxes):
            bbox_tensor[i] = torch.tensor(normalized_boxes[word_idx])
        else:
            bbox_tensor[i] = torch.tensor([0, 0, 0, 0])
    bbox_tensor = bbox_tensor.to(Config.device)
    

    
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            bbox=bbox_tensor.unsqueeze(0),
            attention_mask=attention_mask,
            pixel_values=pixel_values
        )
        predictions = torch.argmax(outputs.logits, dim=-1)[0]
    
    # 解码标签
    word_ids = encoding.word_ids()
    ner_tags = ["O"] * len(word_ids)

    print("ner_tags lenth",len(words))
    for i, wid in enumerate(word_ids):
        if wid is not None and wid < len(words):
            ner_tags[wid] = Config.label_list[predictions[i]]
    
    print(ner_tags)
    print(f"🎯 NER完成: {len(ner_tags)} 个标签")
    print(f"🏷️  示例: {ner_tags[:3]}")

    unified_tags = []
    len1=0
    for i, box in enumerate(text_word_boxes):
        # 获取框内所有单词的标签
        box_tags = [ner_tags[len1 + j] for j in range(len(box))]
        len1 += len(box)
        # 只提取实体类型，忽略 B- 和 I- 前缀
        box_tags_entity = [tag.split('-')[-1] for tag in box_tags if tag != "O"]  # 获取类型部分

        if not box_tags_entity:
            unified_tags.extend(["O"] * len(box_tags))  # 如果没有有效标签，标记为 "O"
            continue
        
        # 统计框内标签的频次（仅统计实体类型，不区分 B- 和 I-）
        label_counts = Counter(box_tags_entity)
        most_common_tag = label_counts.most_common(1)[0][0]  # 获取出现次数最多的实体类型
        
        # 将框内所有单词的标签统一为该实体类型并加上 "I-" 前缀
        unified_tags.extend([f"I-{most_common_tag}"] * len(box_tags))

    print(unified_tags)
    return unified_tags , ner_tags

# box_tags = [ner_tags[j] for j in range(len(ner_tags)) if word_ids[j] in box]


# ===== 5. 🔥 完美可视化=====
def visualize_and_connect(img, words, bboxes, ner_tags ,draw_tags):
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()
    connected_text = ""
    colors = {"HEADER": "blue", "QUESTION": "green", "ANSWER": "yellow"}

    # for box in bboxes:
    #     x1 = box[0]
    #     y1 = box[1]
    #     x2 = box[2]
    #     y2 = box[3]
    #     draw.rectangle([x1, y1, x2, y2], outline="red", width=1)
    #     img.save(Config.output_img1)    
    
    colors = {"HEADER": (0, 0, 255, 50),      # 蓝色半透明
              "QUESTION": (0, 255, 0, 50),    # 绿色半透明
              "ANSWER": (255, 0, 0, 50)}      # 红色半透明
    draw1 = ImageDraw.Draw(img, "RGBA")

    for box, tag in zip(bboxes, draw_tags):
        color = colors.get(tag[2:], (0, 0, 0, 50))  # 默认黑色半透明
        x1, y1, x2, y2 = box
        draw1.rectangle([x1, y1, x2, y2], fill=color)
    img.save(Config.output_img1)    

    for i, (word, tag, box) in enumerate(zip(words, ner_tags, bboxes)):
        # 画红色边框
        img_w, img_h = img.size
        x1 = box[0]
        y1 = box[1]
        x2 = box[2]
        y2 = box[3]
        draw.rectangle([x1, y1, x2, y2], outline="red", width=1)
        # img.save(Config.output_img1)    
        
        # 彩色标签
        color = colors.get(tag[2:], "black")
        label_text = f"{word}\n[{tag}]"
        draw.text((box[0], box[1]), label_text, fill=color, font=font)
        
        # 连贯文本
        if tag.startswith("B-"):
            connected_text += f"\n{tag[2:]}: {word} "
        elif tag.startswith("I-"):
            connected_text += f"{word} "
        else:
            connected_text += f"{word} "
        print(i)
    # 保存图片
    img.save(Config.output_img)
    
    # 保存JSON
    results = {
        "words": words,
        "bboxes": bboxes,
        "ner_tags": ner_tags,
        "connected_text": connected_text.strip()
    }
    with open(Config.output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    
    print(f"✅ 标注图片: {Config.output_img}")
    print(f"✅ JSON结果: {Config.output_json}")
    print(f"📄 连贯NER文本:\n{connected_text.strip()}")



# ===== 6. 主程序=====
if __name__ == "__main__":
    print("🚀" + "="*50)
    print("🎯 PaddleOCR + LayoutLMv3 NER 终极完整版！")
    print("🚀" + "="*50)
    os.makedirs("./output", exist_ok=True)
    os.makedirs("./output1", exist_ok=True)


    # 加载模型
    tokenizer, image_processor, model = load_model()
    
    # 加载图片
    img = Image.open(Config.image_path).convert("RGB")
    print(f"🖼️  图片尺寸: {img.size}")
    
    #预处理图片
    rotate_Image(Config.image_path)

    # OCR识别
    print("\n🔍 阶段1: PaddleOCR识别...")
    words, bboxes , text_word_boxes = run_ocr(Config.processored_img)
    print(len(words))
    print(len(bboxes))

    img = Image.open(Config.processored_img).convert("RGB")
    
    # NER预测
    print("\n🎯 阶段2: LayoutLMv3 NER...")
    if words:
        draw_tags , ner_tags = predict_ner(tokenizer, image_processor, model, img, words, bboxes,text_word_boxes)
        print(len(ner_tags))
        print("\n🖼️  阶段3: 生成可视化...")
        visualize_and_connect(img, words, bboxes, ner_tags, draw_tags)
    else:
        print("⚠️  OCR无结果，保存原图")
        img.save("no_ocr.png")
    
    print("\n🎉" + "="*50)
    print("✅ 任务完成！检查输出文件：")
    print(f"   📸 {Config.output_img}")
    print(f"   📄 {Config.output_json}")
    print("🎉" + "="*50)