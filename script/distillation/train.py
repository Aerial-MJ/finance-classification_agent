import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForLanguageModeling, Trainer, \
    TrainingArguments
from textbrewer import GeneralDistiller, TrainingConfig, DistillationConfig
from datasets import load_dataset
import logging
from Agent.configs.parse import args

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 获取当前脚本文件的绝对路径
current_script_path = os.path.abspath(__file__)
logger.info(f"Current script path: {current_script_path}")

# 获取当前脚本文件所在的目录
current_script_dir = os.path.dirname(current_script_path)
logger.info(f"Current script directory: {current_script_dir}")

# 加载教师模型（DeepSeek-R1:1.5B）
teacher_model_name = args.distill_teacher_model
logger.info(f"Loading teacher model: {teacher_model_name}")
teacher_tokenizer = AutoTokenizer.from_pretrained(teacher_model_name,
    local_files_only=True
)
teacher_model = AutoModelForCausalLM.from_pretrained(teacher_model_name,
    local_files_only=True
)

# 加载学生模型（Qwen）
student_model_name = args.distill_student_model
logger.info(f"Loading student model: {student_model_name}")
student_tokenizer = AutoTokenizer.from_pretrained(student_model_name,
    local_files_only=True
)
student_model = AutoModelForCausalLM.from_pretrained(student_model_name,
    local_files_only=True
)

# 准备数据集
datasets_name = "./wikitext/wikitext-2-raw-v1"  # 确保路径正确

data_files = {
    "train": datasets_name + "/train-00000-of-00001.parquet",  # 修改为正确的路径
    "test": datasets_name + "/test-00000-of-00001.parquet",  # 修改为正确的路径
}

logger.info(f"Loading dataset from local files: {data_files}")

# 因为文件是 Parquet 格式，所以要用 'parquet' 来加载数据
dataset = load_dataset("parquet", data_files=data_files)  # 使用 'parquet' 而不是 'text'

train_dataset = dataset["train"]
eval_dataset = dataset["test"]


# 数据预处理
logger.info(f"Preprocess_function")
def preprocess_function(examples):
    return teacher_tokenizer(examples["text"], truncation=True, padding="max_length", max_length=512)


logger.info("Preprocessing train dataset")
train_dataset = train_dataset.map(preprocess_function, batched=True)
logger.info("Preprocessing eval dataset")
eval_dataset = eval_dataset.map(preprocess_function, batched=True)

# 数据收集器
logger.info("DataCollatorForLanguageModeling")
data_collator = DataCollatorForLanguageModeling(tokenizer=teacher_tokenizer, mlm=False)

# 定义训练参数
logger.info("Creating trainer")
training_args = TrainingArguments(
    output_dir="./results",            # 训练结果保存路径
    eval_strategy="epoch",             # 每个 epoch 结束时评估
    learning_rate=5e-5,                # 学习率（默认 5e-5 是常见选择）
    per_device_train_batch_size=2,     # 每个设备的训练 batch size（GPU 单卡）
    per_device_eval_batch_size=2,      # 每个设备的评估 batch size
    num_train_epochs=3,                # 训练轮次（3 轮可能较短，需根据任务调整）
    weight_decay=0.01,                 # 权重衰减（L2 正则化）
    logging_dir="./logs",              # 日志保存路径
    logging_steps=100,                 # 每 100 步记录一次日志
    fp16=False,                        # 是否启用混合精度训练（建议开启）
    gradient_accumulation_steps=4,     # 梯度累积步数（等效 batch_size=8）
    report_to="tensorboard",           # 使用 TensorBoard 记录训练过程
    # tensorboard_dir="./tensorboard"  # 可选：指定 TensorBoard 日志目录
)

# 定义蒸馏配置  weight:添加权重，"loss": "mse"
logger.info("Creating distillation config")
distill_config = DistillationConfig(
    temperature=2.0,  # 温度参数，控制软标签的平滑程度
    hard_label_weight=0.5,  # 真实标签损失权重
    kd_loss_type="ce",      # 知识蒸馏损失类型（交叉熵）
    intermediate_matches=[  # 中间层匹配配置
        {
            "layer_T": 6,    # 教师模型的第6层
            "layer_S": 6,    # 学生模型的第6层
            "feature": "hidden",  # 匹配隐藏层特征
            "weight": 1.0,   # 中间层损失权重
            "loss": "mse"    # 使用均方误差损失
        }
    ]
)

# 定义训练配置
logger.info("Creating training config")
train_config = TrainingConfig(
    device="cuda" if torch.cuda.is_available() else "cpu",  # 设备选择
    log_dir="./logs",                                     # 日志目录
    output_dir="./outputs"                                # 模型输出目录
    # save_best_model=True,  # 是否保存最佳模型（注释状态）
    # save_last_model=True,  # 是否保存最后模型（注释状态）
    # save_model_every_epoch=True,  # 是否每轮保存模型（注释状态）
    # tensorboard_dir="./tensorboard"  # TensorBoard 日志目录（注释状态）
)

# 创建蒸馏器
logger.info("Creating distiller")
distiller = GeneralDistiller(
    train_config=train_config,        # 训练配置（包含设备、路径等）
    distill_config=distill_config,    # 蒸馏配置（温度、损失权重等）
    model_T=teacher_model,            # 教师模型
    model_S=student_model,            # 学生模型
    adaptor_T=None,                   # 教师模型适配器（未配置）
    adaptor_S=None                    # 学生模型适配器（未配置）
)

# 开始蒸馏
with distiller:  # 使用蒸馏器上下文管理器，确保资源正确初始化和释放
    logger.info("Starting training")  # 记录训练开始日志

    # 初始化 Trainer，集成模型蒸馏配置
    trainer = Trainer(
        model=student_model,  # 学生模型（需要训练的小模型）
        args=training_args,  # 训练参数（如学习率、批次大小、设备等）
        train_dataset=train_dataset,  # 训练数据集（包含输入和标签）
        eval_dataset=eval_dataset,  # 验证数据集（用于评估模型性能）
        data_collator=data_collator,  # 数据批量处理函数（将单条数据组合成批次）
        # processing_class=teacher_tokenizer  # 注意：此处可能存在问题（见下方说明）
        # 正确做法：适配器或数据处理逻辑应在蒸馏配置中处理
    )

    # 开始模型训练
    trainer.train()  # 启动训练循环，包含前向传播、损失计算、反向传播等
    trainer.save_model()

    logger.info("Training finished")  # 记录训练结束日志