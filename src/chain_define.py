from collections import Counter
import os
import numpy as np
import torch
import json
from transformers import (
    LayoutLMv3ForTokenClassification, 
    LayoutLMv3ImageProcessor, 
    LayoutLMv3Tokenizer,  # 🔥 改对了！
    AutoTokenizer,
    XLMRobertaTokenizerFast,
    AutoTokenizer, 
    AutoModelForCausalLM, 
    pipeline,
    Qwen2_5_VLForConditionalGeneration,
    AutoTokenizer, 
    AutoProcessor
)
from paddleocr import PaddleOCR , DocPreprocessor
from PIL import Image, ImageDraw, ImageFont
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from qwen_vl_utils import process_vision_info
import os
import argparse
import json
from pathlib import Path
from typing import Dict, List
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from transformers import AutoTokenizer
from Agent.script.classification.model.resnet_bert import FineGrainedResNetTextFusion
from typing import List
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings


os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
current_directory = os.path.dirname(os.path.abspath(__file__))
print(current_directory)


def image_rotate(image_origin_path):   
    class Config:
        image_path = image_origin_path
        processored_img=current_directory + "/result/preprocess_image.jpg"
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        doc_preprocessor = DocPreprocessor()

    doc_preprocessor = DocPreprocessor()
    # 调用 predict() 返回 generator
    results = doc_preprocessor.predict(
        input=Config.image_path,
        use_doc_orientation_classify=True,   
        use_doc_unwarping=True                
    )

    for res in results:
        angle=res.json["res"]["angle"]
        #print(angle)

    img = Image.open(Config.image_path)
    rotated = img.rotate(angle)

    rotated.save(Config.processored_img)
    # print("=============")
    return Config.processored_img

# image_rotate("/data/postgraduates/2024/chenjiarui/Model/Agent/src/test.jpg")



def invoke_orc_model(image_rotate_path):
    if image_rotate_path is None:
        image_rotate_path=current_directory + "/result/preprocess_image.jpg"

    ocr = PaddleOCR(
        use_doc_orientation_classify=True,
        use_doc_unwarping=False,
        use_textline_orientation=True
    )    
    result = ocr.predict(input=image_rotate_path, return_word_box=False)
    for res in result:
        json_path = f"temp_page.json"
        res.save_to_json(json_path)        
        with open(json_path, "r", encoding="utf-8") as f:
            ocr_data = json.load(f)
        os.remove(json_path)
        ocr_text = " ".join(ocr_data.get("rec_texts", []))

    #print(ocr_text)
    return ocr_text
    
# invoke_orc_model("/data/postgraduates/2024/chenjiarui/Model/Agent/src/test.jpg")



def invoke_VLM_model (image_rotate_path) :
    print(image_rotate_path)
    image_rotate_path="/data/postgraduates/2024/chenjiarui/Model/Agent/src/result/preprocess_image.jpg"
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "/data/postgraduates/2024/chenjiarui/Model/Qwen/Qwen2.5-VL-7B-Instruct", torch_dtype="auto" , device_map="auto" )

    processor = AutoProcessor.from_pretrained("/data/postgraduates/2024/chenjiarui/Model/Qwen/Qwen2.5-VL-7B-Instruct")

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image_rotate_path,
                },
                {"type": "text", 
                "text": """
        请分析下图，为了方便判断它属于以下哪种类型的文件，
        我需要你提取该文件中的关键信息，包括文件的关键字段（如：姓名、日期、金额、编号等）、文本内容、相关布局特征（如：表格、段落、标题等）。
        请按以下格式返回输出：
        {
            "file_type": "文件类型（需要你自己判断）,
            "key_fields": {
                "field_1": "字段值",
                "field_2": "字段值",
                ...
            },
            "layout_features": {
                "has_table": true/false,
                "has_title": true/false,
                "title": "标题文本",
                "table_structure": "表格结构描述（如：列名、行数等）"
            },
            "content_summary": "文件内容简要概述"
        }
        """}
            ],
        }
    ]

    # Preparation for inference
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")

    # Inference: Generation of the output
    generated_ids = model.generate(**inputs, max_new_tokens=512)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )

    del model
    torch.cuda.empty_cache()

    def extract_json_from_text(text: str):
        """从文本中提取第一个 {...} 并解析成 JSON"""
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or start >= end:
            # 找不到合法 JSON
            return None
        json_str = text[start:end+1]
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            return None
        
    result = extract_json_from_text(output_text[0])

    print(result)
    
    return result if result is not None else output_text[0]

# invoke_VLM_model("/data/postgraduates/2024/chenjiarui/Model/Agent/src/test.jpg")



def invoke_ocr_layoutLMv3_model(image_rotate_path):
    if not image_rotate_path :
        image_rotate_path=current_directory + "/result/preprocess_image.jpg"
    os.makedirs(current_directory+"/result/ocr_output", exist_ok=True)
    os.makedirs(current_directory+"/result/ner_output", exist_ok=True)

    class Config:
        model_path = "/data/postgraduates/2024/chenjiarui/Model/LayoutLMv3/layoutlmv3-chinese/layoutlmv3-chinese-trained/best_f1_0.8733"  # 你的训练模型
        base_model = "/data/postgraduates/2024/chenjiarui/Model/LayoutLMv3/layoutlmv3-base-chinese"
        image_path = image_rotate_path
        output_img = current_directory+ "/result/ner_output/ner_annotated.png" 
        output_img1 = current_directory+ "/result/ner_output/ner_annotated_color.png" 
        output_json = current_directory+ "/result/ner_output/ner_results.json" 
        label_list = ["O", "B-HEADER", "I-HEADER", "B-QUESTION", "I-QUESTION", "B-ANSWER", "I-ANSWER"]
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def load_model():
        tokenizer = XLMRobertaTokenizerFast.from_pretrained(Config.base_model)
        image_processor = LayoutLMv3ImageProcessor.from_pretrained(Config.base_model, apply_ocr=False)
        
        model = LayoutLMv3ForTokenClassification.from_pretrained(Config.model_path).to(Config.device)
        model.eval()
        return tokenizer, image_processor, model
        
    def run_ocr(image_path):
        ocr = PaddleOCR(
            use_doc_orientation_classify=True,
            use_doc_unwarping=False,
            use_textline_orientation=True
        )
        
        result = ocr.predict(input=image_path, return_word_box=True)
        for res in result:
            res.save_to_img(current_directory+"/result/ocr_output")
            res.save_to_json(current_directory+"/result/ocr_output")
        
        words, bboxes = [], []

        for page_idx, page_res in enumerate(result):
            json_path = f"{current_directory}/temp_page_{page_idx}.json"
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
                for line_idx, (word_list, box_array) in enumerate(zip(text_words, text_word_boxes)):
                    for token, box in zip(word_list, box_array):
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
                rec_texts = ocr_data.get("rec_texts", [])
                rec_boxes = ocr_data.get("rec_boxes", [])
                for text, box in zip(rec_texts, rec_boxes):
                    if len(box) == 4:
                        xmin, ymin, xmax, ymax = box
                    else:
                        xs, ys = [p[0] for p in box], [p[1] for p in box]
                        xmin, ymin, xmax, ymax = min(xs), min(ys), max(xs), max(ys)
                    words.append(text)
                    bboxes.append([int(xmin), int(ymin), int(xmax), int(ymax)])

        return words, bboxes ,text_word_boxes
    
    def predict_ner(tokenizer, image_processor, model, img, words, bboxes , text_word_boxes):
        if not words:
            return []
        
        pixel_values = image_processor(images=[img], return_tensors="pt")["pixel_values"].to(Config.device)
        w, h = img.size
        
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

        #print("ner_tags lenth",len(words))
        
        for i, wid in enumerate(word_ids):
            if wid is not None and wid < len(words):
                ner_tags[wid] = Config.label_list[predictions[i]]

        unified_tags = []
        len1=0
        for i, box in enumerate(text_word_boxes):
            box_tags = [ner_tags[len1 + j] for j in range(len(box)) if len1 + j< len(ner_tags)]
            len1 += len(box)
            box_tags_entity = [tag.split('-')[-1] for tag in box_tags if tag != "O"]    # 获取类型部分

            if not box_tags_entity:
                unified_tags.extend(["O"] * len(box_tags))  # 如果没有有效标签，标记为 "O"
                continue
            
            label_counts = Counter(box_tags_entity)
            most_common_tag = label_counts.most_common(1)[0][0]
            unified_tags.extend([f"I-{most_common_tag}"] * len(box_tags))

        return unified_tags , ner_tags
        
    def visualize_and_connect(img, words, bboxes, ner_tags ,draw_tags):
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
        img.save(Config.output_img)    

        for i, (word, tag, box) in enumerate(zip(words, ner_tags, bboxes)):
            # 画红色边框
            img_w, img_h = img.size
            x1 = box[0]
            y1 = box[1]
            x2 = box[2]
            y2 = box[3]
            draw.rectangle([x1, y1, x2, y2], outline="red", width=1) 
            
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
        img.save(Config.output_img1)
        
        # 保存JSON
        results = {
            "words": words,
            "bboxes": bboxes,
            "ner_tags": ner_tags,
            "connected_text": connected_text.strip()
        }
        with open(Config.output_json, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=4)
        return connected_text.strip()


    # 加载模型
    tokenizer, image_processor, model = load_model()
    img = Image.open(Config.image_path).convert("RGB")
    words, bboxes , text_word_boxes = run_ocr(Config.image_path)

    if words:
        draw_tags , ner_tags = predict_ner(tokenizer, image_processor, model, img, words, bboxes,text_word_boxes)
        connected_text=visualize_and_connect(img, words, bboxes, ner_tags, draw_tags)
    else:
        img.save(current_directory+"/result/no_ocr.png")

    del model
    torch.cuda.empty_cache()
    return connected_text
# invoke_ocr_layoutLMv3_model("/data/postgraduates/2024/chenjiarui/Model/Agent/src/test.jpg")



def invoke_classification_model(image_rotate_path , ocr_text):
    if image_rotate_path is None:
        image_rotate_path=current_directory + "/result/preprocess_image.jpg"
    class Config:
        checkpoint = "/data/postgraduates/2024/chenjiarui/Model/Agent/script/classification/kfold_checkpoints/fold_3_best.pt"
        image = image_rotate_path
        text = ocr_text
        text_model_name = "/data/postgraduates/2024/chenjiarui/Model/Agent/script/classification/bert-base-chinese"
        data_root = "/data/postgraduates/2024/chenjiarui/Model/Agent/script/classification/data_raw"
        text_json = None
        missing_text_fallback=""
        show_top_k=3
    def load_checkpoint(checkpoint_path: str, device: torch.device) -> tuple:
        """Load checkpoint and extract model configuration."""
        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=device)

        if "args" not in checkpoint:
            raise ValueError(f"Checkpoint {checkpoint_path} does not contain 'args' field")

        args = checkpoint["args"]
        state_dict = checkpoint["state_dict"]

        #print(f"Loaded checkpoint from fold {checkpoint.get('fold', 'N/A')}")
        #print(f"Best validation accuracy: {checkpoint.get('acc', 'N/A'):.4f}")
        #print(f"Epoch: {checkpoint.get('epoch', 'N/A')}")

        return state_dict, args


    def build_model(args: Dict, num_classes: int, device: torch.device) -> FineGrainedResNetTextFusion:
        """Build model from configuration."""
        model = FineGrainedResNetTextFusion(
            num_classes=num_classes,
            pretrained_image=args.get("pretrained_image", True),
            dropout=args.get("dropout", 0.3),
            text_model_name= Config.text_model_name,
            text_trainable=args.get("train_text_encoder", False),
            fusion_dim=args.get("fusion_dim", 768),
            max_text_length=args.get("max_text_length", 512),
        )
        return model.to(device)


    def get_image_transform(img_size: int = 224) -> transforms.Compose:
        """Get image preprocessing transform."""
        return transforms.Compose([
            transforms.Resize(int(img_size * 1.15)),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])


    def load_class_names(data_root: str) -> List[str]:
        """Load class names from data directory."""
        raw_dir = os.path.join(data_root, "raw")
        if not os.path.isdir(raw_dir):
            raise FileNotFoundError(f"Expected raw/ directory under {data_root}")

        class_names = sorted([d for d in os.listdir(raw_dir)
                            if os.path.isdir(os.path.join(raw_dir, d))])
        return class_names


    def load_text_mapping(json_path: str) -> Dict[str, str]:
        """Load text mapping from JSON file."""
        if not os.path.isfile(json_path):
            #print(f"Warning: Text mapping file not found at {json_path}")
            return {}

        with open(json_path, "r", encoding="utf-8") as f:
            text_mapping = json.load(f)

        ##print(f"Loaded {len(text_mapping)} text entries from {json_path}")
        return text_mapping


    def lookup_text_for_image(
        image_path: str,
        text_mapping: Dict[str, str],
        data_root: str,
        default_text: str = ""
    ) -> str:
        """Lookup text for an image path using multiple strategies."""
        data_root = os.path.abspath(data_root)
        image_path_abs = os.path.abspath(image_path)

        # Normalize key function
        def normalize_key(key: str) -> str:
            return key.replace("\\", "/")

        # Strategy 1: path relative to data_root
        try:
            rel_to_root = os.path.relpath(image_path_abs, data_root).replace(os.sep, "/")
            rel_to_root_norm = normalize_key(rel_to_root)
            if rel_to_root_norm in text_mapping:
                return text_mapping[rel_to_root_norm]
        except ValueError:
            pass

        # Strategy 2: try "raw/category/filename" pattern
        parts = Path(image_path_abs).parts
        if len(parts) >= 2:
            category_and_file = f"{parts[-2]}/{parts[-1]}"
            alternative_key = f"raw/{category_and_file}"
            alternative_key_norm = normalize_key(alternative_key)
            if alternative_key_norm in text_mapping:
                return text_mapping[alternative_key_norm]

        # Strategy 3: try filename only
        filename = os.path.basename(image_path)
        for key, value in text_mapping.items():
            if key.endswith(filename):
                return value

        #print(f"Warning: No text found for {image_path}, using default: '{default_text}'")
        return default_text


    @torch.no_grad()
    def predict_single_sample(
        model: FineGrainedResNetTextFusion,
        image_path: str,
        text: str,
        tokenizer: AutoTokenizer,
        transform: transforms.Compose,
        device: torch.device,
        max_text_length: int = 512,
    ) -> tuple:
        """Predict on a single image-text pair."""
        model.eval()

        # Load and preprocess image
        image = Image.open(image_path).convert("RGB")
        image_tensor = transform(image).unsqueeze(0).to(device)

        # Tokenize text
        text_inputs = tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=max_text_length,
            return_tensors="pt"
        )
        text_inputs = {k: v.to(device) for k, v in text_inputs.items()}

        # Prepare text kwargs
        text_kwargs = {"input_ids": text_inputs["input_ids"]}
        if "attention_mask" in text_inputs:
            text_kwargs["attention_mask"] = text_inputs["attention_mask"]
        if "token_type_ids" in text_inputs:
            text_kwargs["token_type_ids"] = text_inputs["token_type_ids"]

        # Forward pass
        logits = model(image_tensor, **text_kwargs)
        probs = F.softmax(logits, dim=1)

        pred_idx = logits.argmax(dim=1).item()
        confidence = probs[0, pred_idx].item()

        return pred_idx, confidence, probs[0].cpu().numpy()

    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #print(f"Using device: {device}")

    # Load checkpoint
    #print(f"\nLoading checkpoint from {Config.checkpoint}...")
    state_dict, train_args = load_checkpoint(Config.checkpoint, device)

    # Load class names
    #print(f"\nLoading class names from {Config.data_root}...")
    class_names = load_class_names(Config.data_root)
    num_classes = len(class_names)
    #print(f"Found {num_classes} classes:")
    # for idx, name in enumerate(class_names):
        #print(f"  {idx}: {name}")

    # Build model
    #print("\nBuilding model...")
    model = build_model(train_args, num_classes, device)
    model.load_state_dict(state_dict)
    model.eval()
    #print("Model loaded successfully")

    # Load tokenizer
    text_model_name = Config.text_model_name
    #print(f"\nLoading tokenizer from {text_model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(text_model_name)

    # Get image transform
    img_size = train_args.get("img_size", 224)
    transform = get_image_transform(img_size)

    # Determine text
    if Config.text is not None:
        text = Config.text
        #print(f"\nUsing provided text: {text}")
    else:
        # Load text mapping and lookup
        text_json = Config.text_json or os.path.join(Config.data_root, "text_raw.json")
        #print(f"\nLoading text mapping from {text_json}...")
        text_mapping = load_text_mapping(text_json)

        text = lookup_text_for_image(
            Config.image,
            text_mapping,
            Config.data_root,
            Config.missing_text_fallback
        )
        #print(f"Looked up text: {text}")

    # Predict
    #print(f"\nRunning inference on {Config.image}...")
    pred_idx, confidence, all_probs = predict_single_sample(
        model,
        Config.image,
        text,
        tokenizer,
        transform,
        device,
        max_text_length=train_args.get("max_text_length", 512)
    )

    # Display results
    # print("\n" + "=" * 80)
    # print("PREDICTION RESULTS")
    # print("=" * 80)
    # print(f"Image: {Config.image}")
    # print(f"Text: {text}")
    # print(f"\nPredicted Class: {class_names[pred_idx]} (index: {pred_idx})")
    # print(f"Confidence: {confidence:.4f} ({confidence * 100:.2f}%)")

    # # Show top-k predictions
    # print(f"\nTop-{Config.show_top_k} Predictions:")
    top_k_indices = all_probs.argsort()[::-1][:Config.show_top_k]

    classification_result=[]
    for rank, idx in enumerate(top_k_indices, 1):
        prob = all_probs[idx]
        # print(f"  {rank}. {class_names[idx]:<30} {prob:.4f} ({prob * 100:.2f}%)")
        classification_result.append({class_names[idx]: f"{prob * 100:.2f}"})

    # print("=" * 80)

    del model
    
    torch.cuda.empty_cache()

    return classification_result
# invoke_classification_model("/data/postgraduates/2024/chenjiarui/Model/Agent/src/test.jpg","ocr识别到的文字")



def invoke_rag_model(rag_text , source = None):
    PERSIST_DIR = "/data/postgraduates/2024/chenjiarui/Model/Agent/script/rag/data/chroma_db"

    EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-0.6B"
    BASE_URL = "https://api.siliconflow.cn/v1"
    API_KEY = "sk-tgprnspwkhliprfcuobqpfiiwjawxkgaldpfkjtovpfudpmf"

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

    print(f"向量库加载成功！共 {vectorstore._collection.count()} 条记录\n")


    # ==================== 测试查询 ====================
    queries = [
        rag_text
    ]

    if source is not None:
        results = []
        for i, query in enumerate(queries, 1):
            docs = vectorstore.similarity_search(query, k=3)
            
            if not docs:
                #print("未检索到任何结果")
                continue

            for j, doc in enumerate(docs, 1):
                
                content = doc.page_content.strip()
                preview = content[:300] + ("..." if len(content) > 300 else "")
                preview = preview.replace("\n", " ").replace("  ", " ")

                source = doc.metadata.get('source', 'unknown')
                line = doc.metadata.get('line', '?')
                label = doc.metadata.get('label', 'N/A')

                # key_fields 处理
                kf = doc.metadata.get('key_fields', {})
                if isinstance(kf, str):
                    try:
                        kf = json.loads(kf)
                    except:
                        kf = {}
                key_preview = " | ".join(f"{k}:{v}" for k, v in kf.items() if v)

                results.append({
                    "id": j,
                    "title": f"[{label}] {source.split('/')[-1]}",  # 智能标题
                    "snippet": preview,
                    "source": source,
                    "line": line,
                    "label": label,
                    "key_fields": kf,
                    "key_preview": key_preview
                })
            return results
    else:
        retrieve_content=""

        for i, query in enumerate(queries, 1):
            #print(f"\n查询 {i}: {query}")
            #print("-" * 50)
            
            # 关键：执行检索，返回 Document 列表
            docs = vectorstore.similarity_search(query, k=4)
            
            if not docs:
                #print("未检索到任何结果")
                continue

            
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

        return retrieve_content
# invoke_rag_model("转账支票-处理")



def invoke_deepseek_model(text):
    BASE_URL = "https://api.deepseek.com"
    API_KEY = "sk-9fc40e8ded4a45f5b9fc61b3330074d3"

    deepseek_chat_model = "deepseek-chat"

    llm1 = ChatOpenAI(model=deepseek_chat_model, api_key=API_KEY, base_url=BASE_URL)


    ans = llm1.invoke([HumanMessage(text)])

    #print(ans.content)

    return ans.content
# invoke_deepseek_model("转账支票-处理")



def invoke_model(rotate_image):
    print("="*60)
    print("图片预处理并识别ocr文字")


    ocr_text = invoke_orc_model(rotate_image)
    print("="*60)
    print("执行分类模型")

    classification_result = invoke_classification_model(rotate_image,ocr_text)
    # print(classification_result)
    for idx , res in enumerate(classification_result) :
        key , value = list(res.keys())[0],list(res.values())[0]
        print(f"  {idx+1}. {key:<30} {float(value)/100:.4f} ({value}%)")

    cl = classification_result[0]
    class_ , score =list(cl.keys())[0] , list(cl.values())[0]

    if(float(score)>90):
        print("当前置信度较高，直接分类为", class_)
        invoke_ocr_layoutLMv3_model(rotate_image)
        return class_
    else:
        print("当前置信度较低，建议进一步调用模型,可能不属于该分类")

    classification_text = "分类模型预测结果（按置信度排序）:\n"
    for idx, res in enumerate(classification_result):
        key, value = list(res.items())[0]
        classification_text += f"{idx+1}. {key} ({value}%)\n"
    print("="*60)
    print("执行ocr——ner模型")
    ocr_json = invoke_ocr_layoutLMv3_model(rotate_image)


    print("="*60)
    print("执行vlm模型")
    VLM_json = invoke_VLM_model(rotate_image)
    # VLM_json = "{'file_type': '健康确认表', 'key_fields': {'姓名': '李秀', '性别': '女', '出生日期': '1982.4.13', '国籍': '中国', '联系电话': '15195851025', '护照号码': '568924', '有效签证': '有', '现居住地址': '成都市东湖国际东光-琉璃路299号', '代理人姓名': '', '代理人证件及号码': '', '填表日期': '', '申请人签名': '', '经办人签名': '', '审核人签署': ''}, 'layout_features': {'has_table': True, 'has_title': True, 'title': '健在确认表（存根）', 'table_structure': '多列多行表格，包含姓名、性别、出生日期、国籍、联系电话、提交证件情况、现居住地址、代理人情况、填表日期、申请人签名、经办人签名、审核人签署等字段'}, 'content_summary': '该文件是一份健康确认表，用于记录个人的基本信息、联系方式、证件情况以及居住地址等。表格中包含了姓名、性别、出生日期、国籍、联系电话、提交证件情况、现居住地址、代理人情况、填表日期、申请人签名、经办人签名、审核人签署等字段。'}"

    print("="*60)
    print("执行rag模型")
    rag_text = invoke_rag_model(str(VLM_json))

    print("="*60)
    print("执行最终的判断模型")    
    final_prompt = (
        "你是一个文档分类模型，需要根据以下信息判断该文档属于哪一类：\n\n"
        f"1. VLM 模型提取的关键信息和布局特征：\n{VLM_json}\n\n"
        f"2. OCR 与 NER 提取的文本信息和命名实体信息：\n{ocr_json}\n\n，请着重注意HEADER标签，是很清晰的分类依据"
        f"3. 检索到的相关背景知识（可以作为参考）：\n{rag_text}\n\n"
        f"4. 先前分类模型的预测结果（因为置信度较低，所以没有直接作为判断依据）：\n{classification_text}\n\n"
        "文档可能属于以下分类之一："
        "业务委托书-处理, 利润表-处理, 特种转账借方-处理, 特种转账贷方-处理, "
        "营业执照-处理, 资产负债表--处理, 身份证反面, 身份证正面--处理, "
        "转账支票-处理, 进账单-处理，也有可能属于其他未列出的分类。\n"
        "请输出文档的最终分类名称，只输出分类，不要其他解释。"
    )

    result=invoke_deepseek_model(final_prompt)

    return result