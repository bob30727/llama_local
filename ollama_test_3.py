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
When responding, there are two things to do:

1. Indicate which type of body gesture this sentence corresponds to.

-Positive Expressions
-Negative Expressions
-Emphasizing Gestures
-Interactive Gestures
-Thinking Gestures
-Anxious/Defensive Gestures

2. Select the most appropriate body gesture from the list below that best matches the meaning of this sentence.

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

Enclose the identified category in 【】. Output only the category. choose one type only. shch as:
【Positive Expressions】【Nodding】
or
【Negative Expressions】【Frowning】

"""},
    {"role": "user", "content": "I'd be happy to introduce you to our latest mobile phone models."}
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

    end_time = time.time()
    print("\n")
    print(f"Execution time: {end_time - start_time} seconds")
else:
    print("Error:", response.status_code, response.text)