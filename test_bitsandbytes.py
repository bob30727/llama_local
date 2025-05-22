# from transformers import AutoTokenizer, AutoModelForCausalLM
# import torch
#
# # model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct-quantized"
# # model_id = "meta-llama/Llama-3.1-8B"
# model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct-AWQ-INT4"
#
#
# # 初始化模型和 tokenizer
# tokenizer = AutoTokenizer.from_pretrained(model_id)
# model = AutoModelForCausalLM.from_pretrained(
#     model_id,
#     device_map="auto",
#     torch_dtype=torch.float16,
#     load_in_8bit=True  # 使用 bitsandbytes 進行低精度推理
# )
#
# # 設置對話
# messages = [
#     {"role": "system", "content": "You are a pirate chatbot who always responds in pirate speak!"},
#     {"role": "user", "content": "Who are you?"},
# ]
#
# # 將對話轉換為輸入文本
# input_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
# inputs = tokenizer(input_text, return_tensors="pt").to("cuda")
#
# # 模型生成
# outputs = model.generate(inputs["input_ids"], max_new_tokens=256, temperature=0.6, top_p=0.9)
#
# # 解碼輸出
# generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
# print(generated_text)


# import bitsandbytes as bnb
# print(bnb.__version__)


# from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
#
# # model_id = "nmeta-llama/Meta-Llama-3.1-8B-Instruct-quantized"
# model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct-AWQ-INT4"
#
# bnb_config = BitsAndBytesConfig(load_in_8bit=True)
#
# model = AutoModelForCausalLM.from_pretrained(
#     model_id,
#     quantization_config=bnb_config,
#     device_map="auto"
# )
#
# tokenizer = AutoTokenizer.from_pretrained(model_id)
#
# inputs = tokenizer("Hello, world!", return_tensors="pt").to("cuda")
# outputs = model.generate(inputs["input_ids"], max_new_tokens=50)
# print(tokenizer.decode(outputs[0], skip_special_tokens=True))


import time
from transformers import AutoModelForCausalLM, AutoTokenizer

# model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct-quantized"
# model_id = "meta-llama/Llama-3.1-8B"
model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct-AWQ-INT4"

# 正確的 rope_scaling 配置
rope_scaling = {
    "type": "linear",  # 或者 "dynamic"
    "factor": 8.0      # 自定義的縮放因子
}

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    rope_scaling=rope_scaling,  # 傳遞正確的配置
    device_map="auto"
)

tokenizer = AutoTokenizer.from_pretrained(model_id)

inputs = tokenizer("Hello, world!", return_tensors="pt").to("cuda")
start_time = time.time()
outputs = model.generate(inputs["input_ids"], max_new_tokens=50)
end_time = time.time()
print(f"Execution time: {end_time - start_time} seconds")

print(tokenizer.decode(outputs[0], skip_special_tokens=True))