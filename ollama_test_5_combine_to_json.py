import requests
import json
import time
import re

# 設定 Ollama API 的端點
OLLAMA_API_URL = "http://localhost:11434/api/chat"

# 指定模型名稱
# model_name = "llama3.2:1b"
model_name = "llama3.1"

sentence = "Its advanced battery life also means you can use it all day without needing to recharge."
# Its advanced battery life also means you can use it all day without needing to recharge.
# we have model A, which packs a powerful processor and plenty of storage space for all your apps and files.
# Next up is model B, which offers a unique blend of style and functionality.

messages = [
    {"role": "system", "content": """
Analyze the given sentence, infer the speaker's intent, and determine the most appropriate body language.  

Step 1: Mark verbs in the sentence using [@timestamp].  
For example: 
"I'd be happy to [@timestamp] introduce [@timestamp] you to our latest mobile phone models."
"we have model A, which [@timestamp] packs [@timestamp] a powerful processor and plenty of storage space for all your apps and files."
"Its advanced battery life also [@timestamp] means [@timestamp] you can use it all day without needing to recharge."
No need for quotation marks. Output format: `###STEP1### <modified sentence>`

Step 2: Determine whether the action in the sentence is positive, negative, or neutral.  
Output format: `###STEP2### <positive/negative/neutral>`

Step 3: Identify whether the sentence's intent is introducing product, casual conversation, or not suitable for action.
(You don’t need to output this step separately.)

Step 4: Based on the result of Step 3, select the most appropriate body gesture from the list below.

**Introducing product:**
- Palm-up gesture
- Pointing at the product
- Slightly open arms
- Leaning slightly forward
- Nodding  

**Casual conversation:**
- Relaxed smiling
- Subtle nodding
- Casual hand gestures
- Hands loosely clasped
- Slight body tilt  

**not suitable for action:**
-none

only output body gesture. Output format: `###STEP4### <selected body gesture>`
"""},
    {"role": "user", "content": sentence}
]

start_time = time.time()

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
    contents = []  # 用來存儲所有內容
    for line in response.iter_lines(decode_unicode=True):
        if line.strip():  # 確保每行有內容
            try:
                data = json.loads(line)  # 將每行 JSON 解析為字典
                content = data.get("message", {}).get("content", "")
                if content:
                    # print(content, end="")  # 按順序顯示回應內容
                    contents.append(content) # 存到列表
            except json.JSONDecodeError as e:
                print(f"\nJSON 解碼錯誤: {e}")
    formatted_text = "".join(contents)

    # **使用正則表達式擷取資訊**
    text_match = re.search(r'###STEP1###\s*(.*?)\s*(?=###STEP2###)', formatted_text, re.DOTALL)
    intention_match = re.search(r'###STEP2###\s*(.*?)\s*(?=###STEP3###|###STEP4###|$)', formatted_text, re.DOTALL)
    motion_match = re.search(r'###STEP4###\s*(.*)', formatted_text, re.DOTALL)

    text = text_match.group(1).strip() if text_match else ""
    intention = intention_match.group(1).strip() if intention_match else "neutral"
    motion = motion_match.group(1).strip() if motion_match else "None"

    # **轉換為 JSON 格式**
    output_json = {
        "text": text,
        "intention": intention,
        "motion_tag": [{"ID": 1, "motion": motion}]
    }

    # **輸出 JSON 結果**
    print(json.dumps(output_json, ensure_ascii=False, indent=2))

    end_time = time.time()
    print("\n")
    print(f"Execution time: {end_time - start_time} seconds")
else:
    print("Error:", response.status_code, response.text)