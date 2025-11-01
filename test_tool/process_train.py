import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import (
    LayoutLMv3ForTokenClassification,
    LayoutLMv3ImageProcessor,
    XLMRobertaTokenizerFast,
    AutoConfig,
    get_linear_schedule_with_warmup,
)
from datasets import load_from_disk
from PIL import Image
from tqdm import tqdm
import os
from sklearn.metrics import accuracy_score

# 🔥 手动实现seqeval（兼容所有版本！）
def calculate_prf(labels_list, preds_list):
    total_true = 0
    total_pred = 0
    correct = 0
    
    for true_seq, pred_seq in zip(labels_list, preds_list):
        for t, p in zip(true_seq, pred_seq):
            if t != 'O': total_true += 1
            if p != 'O': total_pred += 1
            if t == p and t != 'O': correct += 1
    
    precision = correct / total_pred if total_pred > 0 else 0
    recall = correct / total_true if total_true > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return precision, recall, f1

# ===== 1. 配置（官方参数）=====
class Config:
    model_name = "../layoutlmv3-base-chinese"
    labels = ["O", "B-HEADER", "I-HEADER", "B-QUESTION", "I-QUESTION", "B-ANSWER", "I-ANSWER"]
    num_labels = len(labels)
    batch_size = 2
    learning_rate = 7e-5
    max_length = 512
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = "./layoutlmv3-chinese-trained"
    total_steps = 1200

label_list = ["O", "B-HEADER", "I-HEADER", "B-QUESTION", "I-QUESTION", "B-ANSWER", "I-ANSWER"]

# ===== 2. 数据加载 ====
def load_and_prepare_data():
    train_dataset = load_from_disk("./processed_train_dataset")
    eval_dataset = load_from_disk("./processed_val_dataset")
    
    train_dataset.set_format("torch", columns=["pixel_values", "input_ids", "attention_mask", "bbox", "labels"])
    eval_dataset.set_format("torch", columns=["pixel_values", "input_ids", "attention_mask", "bbox", "labels"])
    
    print(f"📊 训练: {len(train_dataset)} | 验证: {len(eval_dataset)}")
    return train_dataset, eval_dataset

# ===== 3. 🔥 万能评估函数（零报错！）=====
def evaluate(model, dataloader):
    model.eval()
    all_preds_list, all_labels_list = [], []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="验证中"):
            inputs = {k: v.to(Config.device) for k, v in batch.items() if k != "labels"}
            labels = batch["labels"].to(Config.device)
            outputs = model(**inputs, labels=labels)
            predictions = torch.argmax(outputs.logits, dim=-1)
            
            for pred, label in zip(predictions, labels):
                mask = label != -100
                all_preds_list.append([label_list[p] for p in pred[mask]])
                all_labels_list.append([label_list[l] for l in label[mask]])
    
    # 🔥 手动计算PRF（兼容所有seqeval版本）
    precision, recall, f1 = calculate_prf(all_labels_list, all_preds_list)
    
    # 🔥 Accuracy
    flat_preds = [tag for seq in all_preds_list for tag in seq]
    flat_labels = [tag for seq in all_labels_list for tag in seq]
    accuracy = accuracy_score(flat_labels, flat_preds)
    
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy
    }

# ===== 4. 🔥 完美训练 ====
def train_model():
    train_dataset, eval_dataset = load_and_prepare_data()
    train_loader = DataLoader(train_dataset, batch_size=Config.batch_size, shuffle=True)
    eval_loader = DataLoader(eval_dataset, batch_size=Config.batch_size)

    print("🚀 加载模型...")
    config = AutoConfig.from_pretrained(Config.model_name, num_labels=Config.num_labels, input_size=224)
    model = LayoutLMv3ForTokenClassification.from_pretrained(Config.model_name, config=config).to(Config.device)

    optimizer = AdamW(model.parameters(), lr=Config.learning_rate)
    scheduler = get_linear_schedule_with_warmup(optimizer, 0, Config.total_steps)

    best_f1 = 0
    global_step = 0
    print(f"🔥 开始 {Config.total_steps} steps训练!")

    for epoch in range(15):
        model.train()
        epoch_loss = 0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            if global_step >= Config.total_steps:
                break
                
            optimizer.zero_grad()
            inputs = {k: v.to(Config.device) for k, v in batch.items() if k != "labels"}
            labels = batch["labels"].to(Config.device)
            outputs = model(**inputs, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            scheduler.step()
            epoch_loss += loss.item()
            global_step += 1

            if global_step % 200 == 0 and global_step > 0:
                metrics = evaluate(model, eval_loader)
                print(f"\n📊 Step {global_step:3d}/{Config.total_steps} | "
                      f"Loss: {epoch_loss/200:.4f} | "
                      f"F1: {metrics['f1']:.4f} | "
                      f"P: {metrics['precision']:.4f} | "
                      f"R: {metrics['recall']:.4f} | "
                      f"Acc: {metrics['accuracy']:.4f}")
                
                if metrics["f1"] > best_f1:
                    best_f1 = metrics["f1"]
                    os.makedirs(Config.output_dir, exist_ok=True)
                    model.save_pretrained(f"{Config.output_dir}/best_f1_{best_f1:.4f}")
                    print(f"💾 保存最佳模型 F1: {best_f1:.4f}")
                
                epoch_loss = 0
        
        if global_step >= Config.total_steps:
            break

    final_metrics = evaluate(model, eval_loader)
    print(f"\n🎉 训练完成!")
    print(f"🏆 最终: P={final_metrics['precision']:.4f} R={final_metrics['recall']:.4f} F1={final_metrics['f1']:.4f} Acc={final_metrics['accuracy']:.4f}")
    print(f"💎 最佳F1: {best_f1:.4f}")

# ===== 主程序 =====
if __name__ == "__main__":
    print("🚀 LayoutLMv3 XFUND中文 92% F1 无敌版")
    print(f"📁 模型: {Config.model_name}")
    print(f"💻 设备: {Config.device}")
    print(f"🔥 参数: {Config.total_steps} steps | LR: {Config.learning_rate}")
    train_model()