import torch
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import get_peft_model, LoraConfig, TaskType
from datasets import Dataset
from torch.nn import functional as F

# 🚀 **讀取本地 CSV 檔案**
csv_path = "data/dataset.csv"  # 請改成你的 CSV 路徑
df = pd.read_csv(csv_path, names=["label", "text"])  # 解析 CSV，第一欄是 label，第二欄是文本

# **將數據轉換為 HF Dataset**
dataset = Dataset.from_pandas(df)

# **選擇 Llama 3.1 1B 模型**
model_name = "meta-llama/Llama-3.2-1B"

# **下載 tokenizer**
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token  # 解決 padding 問題

# **定義 Tokenization 方法**
def tokenize_function(examples):
    text_with_label = [f"[{label}] {text}" for label, text in zip(examples["label"], examples["text"])]
    return tokenizer(text_with_label, padding="max_length", truncation=True, max_length=128)

# **處理數據**
tokenized_datasets = dataset.map(tokenize_function, batched=True)

# **處理沒有 test 集的情況**
if "test" not in tokenized_datasets:
    print("No test set found, splitting train dataset...")
    tokenized_datasets = tokenized_datasets.train_test_split(test_size=0.1)

# **確保 `eval_dataset` 存在**
eval_dataset = tokenized_datasets["test"] if "test" in tokenized_datasets else None

# **載入模型**
model = AutoModelForCausalLM.from_pretrained(
    model_name, torch_dtype=torch.float16, device_map="auto"
)

# **設定 LoRA 參數**
lora_config = LoraConfig(
    r=8, lora_alpha=32, lora_dropout=0.1,
    bias="none", task_type=TaskType.CAUSAL_LM,
    target_modules=["q_proj", "v_proj"]
)

# **應用 LoRA 微調**
model = get_peft_model(model, lora_config)

# **自訂 Loss 計算**
class MyTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs["input_ids"]  # Llama 是自回歸模型
        outputs = model(**inputs)
        logits = outputs.logits

        # Shift tokens left for next-token prediction
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1), ignore_index=tokenizer.pad_token_id
        )

        return (loss, outputs) if return_outputs else loss

# **訓練參數**
training_args = TrainingArguments(
    output_dir="./llama3_finetuned",
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    logging_dir="./logs",
    logging_steps=10,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-4,
    num_train_epochs=3,
    weight_decay=0.01,
    fp16=True,
    push_to_hub=False
)

# **設定 Trainer**
trainer = MyTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=eval_dataset
)

# **開始微調**
trainer.train()

# **儲存模型**
trainer.save_model("./llama3_finetuned")