# from transformers import LlamaForCausalLM, LlamaTokenizer
# from peft import get_peft_model, LoraConfig, TaskType
# from datasets import load_dataset
# import torch
#
# # 加載模型和分詞器
# model_name = "meta-llama/Llama-3.2-1B"
# model = LlamaForCausalLM.from_pretrained(
#     model_name,
# )
# tokenizer = LlamaTokenizer.from_pretrained(model_name)


# import torch
# import torchvision
#
# print(torch.__version__)       # 確保 Torch 版本一致
# print(torchvision.__version__) # 確保 torchvision 版本匹配


# from datasets import load_dataset
# dataset = load_dataset("Abirate/english_quotes")
#
# # 查看 dataset 的 keys
# print(dataset)
#
# # 查看第一筆數據
# print(dataset['train'][0])


import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import get_peft_model, LoraConfig, TaskType
from datasets import load_dataset
from torch.nn import functional as F

# 自訂 loss 計算
class MyTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs["input_ids"]  # Llama 是自回歸模型，labels 和 inputs 一樣
        outputs = model(**inputs)
        logits = outputs.logits

        # Shift tokens left for next-token prediction
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        # 計算 cross-entropy loss
        loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1), ignore_index=tokenizer.pad_token_id)

        return (loss, outputs) if return_outputs else loss

# 選擇 Llama 3.1 1B 模型
model_name = "meta-llama/Llama-3.2-1B"

# 下載模型和 tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token  # 解決 padding 問題

model = AutoModelForCausalLM.from_pretrained(
    model_name, torch_dtype=torch.float16, device_map="auto"
)

# 設定 LoRA 參數
lora_config = LoraConfig(
    r=8,  # 降維維度
    lora_alpha=32,
    lora_dropout=0.1,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
    target_modules=["q_proj", "v_proj"],  # 適用於 Llama 模型
)

# 應用 LoRA 微調
model = get_peft_model(model, lora_config)

# 載入訓練數據（可用自己的 dataset）
dataset = load_dataset("Abirate/english_quotes")  # 這裡用 quotes dataset 當範例

# 定義 Tokenization 方法
def tokenize_function(examples):
    return tokenizer(
        examples["quote"],
        padding="max_length",
        truncation=True,
        max_length=128
    )

tokenized_datasets = dataset.map(tokenize_function, batched=True)

# **處理沒有 test 集的情況**
if "test" not in tokenized_datasets:
    print("No test set found, splitting train dataset...")
    tokenized_datasets = tokenized_datasets["train"].train_test_split(test_size=0.1)

# **確保 `eval_dataset` 不會出錯**
eval_dataset = tokenized_datasets["test"] if "test" in tokenized_datasets else None

# 訓練參數
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

# # 設定 Trainer
# trainer = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=tokenized_datasets["train"],
#     eval_dataset=eval_dataset,  # **確保 `test` 存在**
#     compute_loss=compute_loss,  # 🔥 指定自訂的 loss function
# )

# 設定 Trainer
trainer = MyTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=eval_dataset,  # **確保 `test` 存在**
)

# 開始微調
trainer.train()

# 儲存模型
trainer.save_model("./llama3_finetuned")