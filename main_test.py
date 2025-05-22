import time
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from accelerate import init_empty_weights
import torch

# model = AutoModelForCausalLM.from_pretrained('Llama2_Chinese_7b_Chat',device_map='auto',torch_dtype=torch.float16,load_in_8bit=True)
# print("___________done__________")
# model =model.eval()
# tokenizer = AutoTokenizer.from_pretrained('Llama2_Chinese_7b_Chat',use_fast=False)
# tokenizer.pad_token = tokenizer.eos_token
# input_ids = tokenizer(['<s>Human: 介绍一下中国\n</s><s>Assistant: '], return_tensors="pt",add_special_tokens=False).input_ids.to('cuda')
# generate_input = {
#     "input_ids":input_ids,
#     "max_new_tokens":512,
#     "do_sample":True,
#     "top_k":50,
#     "top_p":0.95,
#     "temperature":0.3,
#     "repetition_penalty":1.3,
#     "eos_token_id":tokenizer.eos_token_id,
#     "bos_token_id":tokenizer.bos_token_id,
#     "pad_token_id":tokenizer.pad_token_id
# }
# generate_ids  = model.generate(**generate_input)
# text = tokenizer.decode(generate_ids[0])
# print(text)

# 1. 設定模型名稱或本地路徑
# model_name_or_path = "./Llama2_Chinese_7b_Chat"
model_name_or_path = "meta-llama/Llama-3.1-8B"
# model_name_or_path = "meta-llama/Meta-Llama-3.1-8B-Instruct-quantized"
# model_name_or_path = "meta-llama/Meta-Llama-3.1-8B-Instruct-AWQ-INT4"


# 2. 啟用量化（第 5 步）
bnb_config = BitsAndBytesConfig(load_in_8bit=True)  # 使用 8-bit 加速

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"torch device is {DEVICE}")

# 3. 分層載入模型（第 6 步）
print("Initializing model with empty weights...")
with init_empty_weights():
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        # quantization_config=bnb_config,
        device_map="auto"  # 自動分配設備（如多張 GPU）
    )
# model = AutoModelForCausalLM.from_pretrained(model_name_or_path)

# 4. 載入分詞器
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

# 5. 檢查設備
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Model will run on: {device}")

# 6. 測試模型
print("Testing the model...")
prompt = "I have a speling error in this sentence."
correction_prompt = f"Please correct the following text with proper spelling and grammar: '{prompt}'"
input_text = "The quick brown fox jumps over the lazy dog."
inputs = tokenizer(correction_prompt, return_tensors="pt").to(device)
start_time = time.time()
# 推論產生文字
outputs = model.generate(inputs["input_ids"], max_length=50, num_return_sequences=1)
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
end_time = time.time()
print(f"Execution time: {end_time - start_time} seconds")
print("Generated text:")
print(generated_text)

# model = AutoModelForCausalLM.from_pretrained(model_id)

# pipeline = transformers.pipeline(
#     "text-generation", model=model_id, model_kwargs={"torch_dtype": torch.bfloat16}, device_map="auto"
# )
question = input("Question: ")

model = model.to(DEVICE)
input_text = "The quick brown fox"
inputs = tokenizer(input_text, return_tensors="pt").to(DEVICE)

outputs = model.generate(inputs["input_ids"], max_length=50, num_return_sequences=1)
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(generated_text)


# print(pipeline("Hey how are you doing today?"))
# print(f"Execution time: {end_time - start_time} seconds")
# tokenizer = LlamaTokenizer.from_pretrained(model_id)
# model = LlamaForCausalLM.from_pretrained(model_id, device_map='auto', torch_dtype=torch.float16)



# while (1):
#     eval_prompt = input("input:")
#
#     model_input = tokenizer(eval_prompt, return_tensors="pt").to(DEVICE)
#
#     model.eval()
#     with torch.no_grad():
#         print(tokenizer.decode(model.generate(**model_input, max_new_tokens=500)[0], skip_special_tokens=True))