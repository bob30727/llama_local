import requests
import json
import time
import re

# 設定 Ollama API 的端點
OLLAMA_API_URL = "http://localhost:11434/api/chat"

# 指定模型名稱
# model_name = "llama3.2:1b"
model_name = "llama3.1"

# Select the most relevant body gesture from the list below, or none if there is no suitable gesture.
messages = [
    {"role": "system", "content": """
Please act as a salesman. You have three types of mobile phones: A, B, and C. Whenever someone asks about mobile phones, introduce these three models to them.

When responding, there are three things to do:
1. Mark the verbs using markdown syntax in every sentences.
2. Indicate whether the verb is positive or negative or neutral.
3. Select the most appropriate body gesture from the list below that best matches the meaning of this sentence.
 
body gesture:
-Positive Expressions
Nodding
Smiling
Open-hand gestures
Leaning forward

-Negative Expressions
Frowning
Shaking head
Arms crossed
Turning away or leaning back

-Emphasizing Gestures
Hand waving
Pointing
Table tapping
Expanding arms

-Interactive Gestures
Reaching out
Eye contact
Handshaking
Body orientation

-Thinking Gestures
Chin touching
Head scratching
Looking down in thought
Furrowing brows

-Anxious/Defensive Gestures
Self-hugging
Fidgeting
Shuffling feet
Avoiding eye contact

-No Gesture Needed
none

shch as:
I'd be happy to **introduce** (positive) [Pointing] you to our latest mobile phone models. We **have** (neutral) [Smiling] three top-of-the-line options that I think you'll love.
First, we have model A, which **packs** (positive) [none] a powerful processor and plenty of storage space for all your apps and files. It's ideal for heavy users who need a fast and reliable device.

"""},
    {"role": "user", "content": "Can you introduce the mobile phones to me?"}
]

# 發送請求給 Ollama
response = requests.post(
    OLLAMA_API_URL,
    json={
        "model": model_name,
        "messages": messages,
        "temperature" : 0
    },
    stream=True  # 啟用流式處理
)

# 檢查回應狀態碼
if response.status_code == 200:
    print("回覆:")
    # 處理返回的多行 JSON
    start_time = time.time()
    contents = []  # 用來存儲所有內容
    for line in response.iter_lines(decode_unicode=True):
        if line.strip():  # 確保每行有內容
            try:
                data = json.loads(line)  # 將每行 JSON 解析為字典
                content = data.get("message", {}).get("content", "")
                if content:
                    print(content, end="")  # 按順序顯示回應內容
                    contents.append(content) # 存到列表
            except json.JSONDecodeError as e:
                print(f"\nJSON 解碼錯誤: {e}")
    formatted_text = "".join(contents)
    # extracted_texts = re.findall(r'【(.*?)】', formatted_text)  # 半形引號 "
    # result = "\n".join(extracted_texts)
    # with open("response.txt", "w", encoding="utf-8") as file:
    #     file.write(result)

# if response.status_code == 200:
#     print("回覆:")
#     start_time = time.time()
#     contents = []  # 用來存儲所有的 question
#     for line in response.iter_lines(decode_unicode=True):
#         if line.strip():  # 確保每行有內容
#             try:
#                 data = json.loads(line)  # 解析 JSON
#                 content = data.get("message", {}).get("content", "")
#                 if content:
#                     contents.append(content)  # 存到列表
#             except json.JSONDecodeError as e:
#                 print(f"\nJSON 解碼錯誤: {e}")
#     full_json = "".join(contents)
#     print(full_json)
#     match = re.findall(r"```(.*?)```", full_json, re.DOTALL)
#     if match:
#         json_string = match[0]  # 取出第一個匹配到的 JSON 內容
#         try:
#             data = json.loads(json_string)  # 解析 JSON
#
#             # 存入 JSON 檔案
#             with open("response.json", "w", encoding="utf-8") as file:
#                 json.dump(data, file, indent=4, ensure_ascii=False)
#
#             print(f"已成功將數據儲存至 response.json")
#         except json.JSONDecodeError as e:
#             print(f"合併 JSON 解析失敗: {e}")
    # else:
    #     print("未找到 JSON 格式的內容")

    end_time = time.time()
    print("\n")
    print(f"Execution time: {end_time - start_time} seconds")
else:
    print("Error:", response.status_code, response.text)