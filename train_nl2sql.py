"""
NL2SQL 模型微调训练脚本
使用 QLoRA + 4-bit 量化在 Qwen3.5-4B 上进行微调

使用 ModelScope 镜像加速下载
"""

import os
import json
import torch

# 设置 ModelScope 为默认下载源
os.environ["MODELSCOPE_SDK_DOWNLOAD_HOST"] = "https://modelscope.cn"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig
)
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType
)
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 添加 print 作为备用
def log_info(msg):
    print(msg)
    logger.info(msg)

# 替换 HuggingFace 下载
from modelscope import snapshot_download

def download_model(model_id: str, cache_dir: str = "./model_cache"):
    """从 ModelScope 下载模型"""
    print(f"[ModelScope] 下载模型: {model_id}")
    model_dir = snapshot_download(
        model_id,
        cache_dir=cache_dir,
        revision="master"
    )
    return model_dir

# 配置 - 使用本地模型路径
MODEL_ID = "Qwen/Qwen3.5-4B"
LOCAL_MODEL_PATH = "/root/autodl-tmp/models/qwen3.5-4b"  # 本地模型路径

# 显存优化配置
MICRO_BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 8  # 增加梯度累积
BATCH_SIZE = 1  # 减小 batch
USE_8BIT = True  # 使用 8-bit 量化（更省显存）

# 完整配置
OUTPUT_DIR = "./models/nl2sql-qwen3.5-4b"
DATA_PATH = "./data/nl2sql_train.jsonl"
EPOCHS = 3
LEARNING_RATE = 2e-4
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
MAX_SEQ_LENGTH = 256  # 减小序列长度


def format_prompt(example):
    """格式化训练数据为指令格式"""
    prompt = f"""<|im_start|>system
你是一个SQL查询专家，负责将用户的自然语言问题转换为准确的SQL查询语句。
<|im_end|>
<|im_start|>user
根据用户问题生成SQL查询。只返回SQL语句，不要解释。

问题：{example['question']}
<|im_end|>
<|im_start|>assistant
{example['sql']}
<|im_end|>"""
    return {"text": prompt}


def load_and_process_data():
    """加载并处理训练数据"""
    logger.info(f"加载数据 from {DATA_PATH}")
    
    # 读取 JSONL 数据
    data = []
    with open(DATA_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            # 转换为简单的 question -> sql 格式
            data.append({
                "question": item["input"].replace("问题：", ""),
                "sql": item["output"]
            })
    
    logger.info(f"加载了 {len(data)} 条数据")
    
    # 应用格式化
    formatted_data = [format_prompt(d) for d in data]
    
    return formatted_data


def tokenize_function(tokenizer, example, max_length=MAX_SEQ_LENGTH):
    """Tokenize 文本"""
    result = tokenizer(
        example["text"],
        truncation=True,
        max_length=max_length,
        padding="max_length",
        return_tensors=None
    )
    # labels 复制 input_ids
    result["labels"] = result["input_ids"].copy()
    return result


def setup_model():
    """设置模型和量化配置"""
    logger.info(f"加载模型: {MODEL_ID}")
    
    # 使用本地模型路径
    model_path = LOCAL_MODEL_PATH
    
    # 量化配置 - 根据显存选择
    if USE_8BIT:
        # 8-bit 量化 - 更稳定
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,
        )
    else:
        # 4-bit 量化 - 更省显存但可能不稳定
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True
        )
    
    # 加载模型 - 使用 CPU offload
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=bnb_config,
        device_map="cpu",  # 先加载到 CPU
        trust_remote_code=True,
        torch_dtype=torch.float16
    )
    
    # 冻结原始权重
    for param in model.parameters():
        param.requires_grad = False
    
    logger.info("模型加载完成")
    return model


def setup_lora(model):
    """设置 LoRA 配置"""
    logger.info("配置 LoRA")
    
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj", 
            "gate_proj", "up_proj", "down_proj"
        ],
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    return model


def main():
    """主训练流程"""
    logger.info("=" * 50)
    logger.info("NL2SQL 模型微调训练")
    logger.info(f"模型: {MODEL_ID}")
    logger.info("=" * 50)
    
    # 1. 加载数据
    train_data = load_and_process_data()
    
    # 2. 加载 tokenizer
    logger.info("加载 tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        LOCAL_MODEL_PATH,
        trust_remote_code=True,
        padding_side="right"
    )
    tokenizer.pad_token = tokenizer.eos_token
    
    # 3. Tokenize 数据
    logger.info("Tokenizing 数据...")
    tokenized_dataset = []
    for item in train_data:
        tokenized = tokenize_function(tokenizer, item)
        tokenized_dataset.append(tokenized)
    
    # 4. 设置模型
    model = setup_model()
    model = setup_lora(model)
    
    # 5. 训练参数
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=MICRO_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        per_device_eval_batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        num_train_epochs=EPOCHS,
        logging_dir=f"{OUTPUT_DIR}/logs",
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=2,
        warmup_steps=10,
        bf16=True,
        dataloader_num_workers=0,
        remove_unused_columns=False,
        optim="paged_adamw_32bit",
        lr_scheduler_type="cosine",
    )
    
    # 6. Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # Causal LM
    )
    
    # 7. 开始训练
    logger.info("开始训练...")
    from torch.utils.data import DataLoader
    
    train_dataloader = DataLoader(
        tokenized_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=data_collator
    )
    
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=LEARNING_RATE,
        weight_decay=0.01
    )
    
    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        for step, batch in enumerate(train_dataloader):
            # 移动到设备
            batch = {k: v.cuda() if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            # 前向传播
            outputs = model(**batch)
            loss = outputs.loss / GRADIENT_ACCUMULATION_STEPS
            total_loss += loss.item() * GRADIENT_ACCUMULATION_STEPS
            
            # 反向传播
            loss.backward()
            
            if (step + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
                optimizer.step()
                optimizer.zero_grad()
            
            if step % 10 == 0:
                logger.info(f"Epoch {epoch+1}/{EPOCHS}, Step {step}, Loss: {loss.item()*GRADIENT_ACCUMULATION_STEPS:.4f}")
        
        logger.info(f"Epoch {epoch+1} 完成, 平均 Loss: {total_loss/len(train_dataloader):.4f}")
        
        # 保存 checkpoint
        checkpoint_path = f"{OUTPUT_DIR}/checkpoint-epoch{epoch+1}"
        model.save_pretrained(checkpoint_path)
        tokenizer.save_pretrained(checkpoint_path)
        logger.info(f"已保存 checkpoint: {checkpoint_path}")
    
    # 8. 保存最终模型
    final_path = f"{OUTPUT_DIR}/final"
    model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)
    logger.info(f"训练完成！模型保存到: {final_path}")
    
    # 9. 保存训练配置
    config = {
        "model_id": MODEL_ID,
        "output_dir": OUTPUT_DIR,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "lora_r": LORA_R,
        "lora_alpha": LORA_ALPHA,
        "data_path": DATA_PATH,
        "num_samples": len(train_data)
    }
    with open(f"{OUTPUT_DIR}/config.json", 'w') as f:
        json.dump(config, f, indent=2)
    
    logger.info("=" * 50)
    logger.info("训练完成！")
    logger.info("=" * 50)


if __name__ == "__main__":
    main()
