
import os
from PIL import Image
from tqdm import tqdm
import json
import numpy as np
import torch
from transformers import (
    LayoutLMv3ForTokenClassification, 
    LayoutLMv3ImageProcessor, 
    LayoutLMv3Tokenizer, 
    AutoTokenizer,
    XLMRobertaTokenizerFast
)
from PIL import Image, ImageDraw, ImageFont
from paddleocr import PaddleOCR
import json
import os
from tqdm import tqdm
from collections import Counter
from paddleocr import DocPreprocessor
from Agent.configs.parse import args

# ===== 1. 配置 =====
class Config:
    model_path =  args.layoutLMv3_train_model
    base_model =  args.layoutLMv3_base_model  
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



# ===== 预处理图片  =====
def rotate_Image(image_path , output_path):

    os.makedirs(os.path.join(output_path, "output"), exist_ok=True)
    os.makedirs(os.path.join(output_path, "output1"), exist_ok=True)
    
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

    preprocess_img_path=os.path.join(output_path, "output" ,"preprocessed.png")

    rotated.save(preprocess_img_path)

    return preprocess_img_path


# ===== 3. 🔥 完美PaddleOCR=====
def run_ocr(image_path,output_path):
    ocr = PaddleOCR(
        use_doc_orientation_classify=True,
        use_doc_unwarping=False,
        use_textline_orientation=True
    )
    
    result = ocr.predict(input=image_path, return_word_box=True)
    # print(result)


    for res in result:
        # res.print()
        res.save_to_img(os.path.join(output_path, "output1"))
        res.save_to_json(os.path.join(output_path, "output1"))
    
    words, bboxes = [], []

    for page_idx, page_res in enumerate(result):
        # print(f"\n🔍 页面 {page_idx+1}: OCRResult 对象")

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
            # print(f"✅ 检测到 {len(text_words)} 个 token 级文本块")

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
            # print(f"⚠️ 使用行级识别结果 {len(rec_texts)} 条")

            for text, box in zip(rec_texts, rec_boxes):
                if len(box) == 4:
                    xmin, ymin, xmax, ymax = box
                else:
                    xs, ys = [p[0] for p in box], [p[1] for p in box]
                    xmin, ymin, xmax, ymax = min(xs), min(ys), max(xs), max(ys)
                words.append(text)
                bboxes.append([int(xmin), int(ymin), int(xmax), int(ymax)])

        # print(f"✅ OCR识别: {len(words)} 个文本块")
        # print(f"📝 前3个文本: {words[:3]}")
        # print(f"📝 前3个方框: {bboxes[:3]}")

    return words, bboxes ,text_word_boxes


# ===== 4. 🔥 完美NER预测=====
def predict_ner(tokenizer, image_processor, model, img, words, bboxes , text_word_boxes ):
    if not words:
        # print("⚠️  无文本，跳过NER")
        return []
    
    # print(f"🎯 开始NER预测...")
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

    # print("ner_tags lenth",len(words))
    for i, wid in enumerate(word_ids):
        if wid is not None and wid < len(words):
            ner_tags[wid] = Config.label_list[predictions[i]]
    

    unified_tags = []
    len1 = 0
    for i, box in enumerate(text_word_boxes):
        # 获取框内所有单词的标签
        print("box = ",len(box))
        print("len1 = ",len1)
        print("ner_tags",len(ner_tags))
        box_tags = [ner_tags[len1 + j] for j in range(len(box)) if len1 + j < len(ner_tags)]
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

    # print(unified_tags)
    return unified_tags , ner_tags

# box_tags = [ner_tags[j] for j in range(len(ner_tags)) if word_ids[j] in box]


# ===== 5. 🔥 完美可视化=====
def visualize_and_connect(img, words, bboxes, ner_tags ,draw_tags,output_path):
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()
    connected_text = ""
    colors = {"HEADER": "blue", "QUESTION": "green", "ANSWER": "yellow"}

    
    colors = {"HEADER": (0, 0, 255, 50),      # 蓝色半透明
              "QUESTION": (0, 255, 0, 50),    # 绿色半透明
              "ANSWER": (255, 0, 0, 50)}      # 红色半透明
    draw1 = ImageDraw.Draw(img, "RGBA")

    for box, tag in zip(bboxes, draw_tags):
        color = colors.get(tag[2:], (0, 0, 0, 50))  # 默认黑色半透明
        x1, y1, x2, y2 = box
        draw1.rectangle([x1, y1, x2, y2], fill=color)
    img.save(os.path.join(output_path, "output","ner_annotated.png"))    

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
        # print(i)
    # 保存图片
    img.save(os.path.join(output_path, "output","ner_annotated1.png"))
    
    # 保存JSON
    results = {
        "words": words,
        "bboxes": bboxes,
        "ner_tags": ner_tags,
        "connected_text": connected_text.strip()
    }
    with open(os.path.join(output_path, "output","ner_results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)




def process_image(image_path, output_dir, tokenizer, image_processor, model):
    # 获取图片名称，不带扩展名
    image_name = os.path.basename(image_path).split('.')[0]
    
    # 创建存放结果的文件夹
    image_output_dir = os.path.join(output_dir)
    os.makedirs(image_output_dir, exist_ok=True)
    
    # 加载图片
    img = Image.open(image_path).convert("RGB")
    
    preprocess_image=rotate_Image(image_path , output_dir)

    # OCR识别
    print(f"🔍 正在处理 {image_path} 图片的OCR...")
    words, bboxes ,text_word_boxes = run_ocr(preprocess_image,output_dir)
    
    img = Image.open(preprocess_image).convert("RGB")
    # NER预测
    print(f"🎯 正在为 {image_path} 图片进行NER预测...")
    if words:
        draw_tags , ner_tags = predict_ner(tokenizer, image_processor, model, img, words, bboxes,text_word_boxes)
        
        # 可视化并连接标签
        print(f"🖼️ 正在生成 {image_path} 图片的NER结果...")
        visualize_and_connect(img, words, bboxes, ner_tags, draw_tags,output_dir)



# 批量处理所有图片
def batch_process_images(image_folder,  tokenizer, image_processor, model):
    # 获取所有图片文件路径
    image_paths = []
    for subdir, _, files in os.walk(image_folder):
        depth = subdir[len(image_folder):].count(os.sep)
        if depth > 2:  # 超过 2 级就跳过
            continue
        for file in files:
            if file.endswith(".jpg") or file.endswith(".jpeg") or file.endswith(".png"):
                image_paths.append(os.path.join(subdir, file))
    print(image_paths)
    print(len(image_paths))

    # 逐个处理图片
    for image_path in tqdm(image_paths, desc="处理图片"):
        print("处理"+image_path)
        output_dir = os.path.abspath(image_path).split('.')[0]
        # if not os.path.exists(output_dir):
        process_image(image_path, output_dir, tokenizer, image_processor, model)

# 设置图片和输出文件夹路径
image_folder = args.rag_data_dir  # 存放图片的文件夹路径

# 假设你已经加载了模型和tokenizer
tokenizer, image_processor, model = load_model()

# 执行批量处理
batch_process_images(image_folder, tokenizer, image_processor, model)