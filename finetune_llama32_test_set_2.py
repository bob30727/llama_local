import torch
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import get_peft_model, LoraConfig, TaskType
from datasets import Dataset
from torch.nn import functional as F

# 🚀 **讀取本地 CSV 檔案**
train_csv_path = "data/atis_intents_train.csv"  # 你的訓練集
test_csv_path = "data/atis_intents_test.csv"  # 你的測試集

df_train = pd.read_csv(train_csv_path, names=["label", "text"])
df_test = pd.read_csv(test_csv_path, names=["label", "text"])

# 🔥 **建立 `label2id` 映射字典**
unique_labels = df_train["label"].unique().tolist()
label2id = {label: i for i, label in enumerate(unique_labels)}

# **轉換 `label` 為數字**
df_train["label"] = df_train["label"].map(label2id)
df_test["label"] = df_test["label"].map(label2id)

# **轉換為 HF Dataset**
train_dataset = Dataset.from_pandas(df_train)
test_dataset = Dataset.from_pandas(df_test)

# **選擇 Llama 3.1 1B 模型**
model_name = "meta-llama/Llama-3.2-1B"

# **下載 tokenizer**
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token  # 避免 padding 問題

# **定義 Tokenization 方法**
def tokenize_function(examples):
    tokenized = tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=64,
        return_tensors="pt",
    )

    # **處理 labels** (確保 labels 的維度與 input_ids 一致)
    tokenized["labels"] = tokenized["input_ids"].clone()  # 確保 labels 也有對應的 token
    return tokenized

# **處理數據**
tokenized_train = train_dataset.map(tokenize_function, batched=True)
tokenized_test = test_dataset.map(tokenize_function, batched=True)

# **確保 `labels` 欄位為 `int`**
tokenized_train = tokenized_train.map(lambda x: {"labels": x["label"]}, remove_columns=["label"])
tokenized_test = tokenized_test.map(lambda x: {"labels": x["label"]}, remove_columns=["label"])

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
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs["labels"]  # 確保使用正確的 labels
        outputs = model(**inputs)
        logits = outputs.logits

        print(f"logits size: {logits.size()}, labels size: {labels.size()}")

        # Shift tokens left for next-token prediction
        shift_logits = logits[..., :-1, :].contiguous()  # 🟢 修正 batch size 不匹配
        shift_labels = labels[..., 1:].contiguous()  # 🟢 Shift labels 確保 shape 一致

        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=tokenizer.pad_token_id
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
    train_dataset=tokenized_train,
    eval_dataset=tokenized_test  # ✅ 這裡直接用 test set
)

# **開始微調**
trainer.train()

# **儲存模型**
trainer.save_model("./llama3_finetuned")