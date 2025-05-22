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
Analyze the given sentence, infer the speaker's intent, and determine the most appropriate body language.

Step 1:
Mark all verbs in the sentence using [@timestamp].
For example: "I'd be happy to [@timestamp] introduce [@timestamp] you to our latest mobile phone models."

Step 2:
Determine whether the action in the sentence is positive, negative, or neutral.

Step 3:
Identify whether the sentence's intent is introducing product or casual conversation.

Step 4:
Based on the result of Step 3, select the most appropriate body gesture from the list below.

If the intent is "introducing product", choose from:
Palm-up gesture
Pointing at the product
Slightly open arms
Leaning slightly forward
Nodding

If the intent is "casual conversation", choose from:
Relaxed smiling
Subtle nodding
Casual hand gestures
Hands loosely clasped
Slight body tilt

Display the results of Step 1, Step 2, Step 3, and Step 4 in JSON format. Example Output:
{
  "text": "I'd be happy to [@timestamp] introduce [@timestamp] you to our latest mobile phone models.",
  "intension": "positive",
  "motion_tag": [
    {
      "ID": 1,
      "motion": "Pointing at the product"
    }
  ]
}

"""},
    {"role": "user", "content": "Its advanced battery life also means you can use it all day without needing to recharge."}
]
# we have model A, which packs a powerful processor and plenty of storage space for all your apps and files.
# Its advanced battery life also means you can use it all day without needing to recharge.

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