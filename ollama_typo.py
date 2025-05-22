import requests
import json
import time
import re

# 設定 Ollama API 的端點
OLLAMA_API_URL = "http://localhost:11434/api/chat"

# 指定模型名稱
# model_name = "llama3.2:1b"
model_name = "llama3.1"

messages = [
    {"role": "system", "content": """
Please correct the following text with proper spelling and grammare. The corrected sentences enclosed in specific symbols 【】.

Here are some proper nouns for your reference:
MediaMarktSaturn,
google,
Apple,

"""},
    {"role": "user", "content": "I want to buy something in MediaMarketSaturn"}
]

# Please correct the following text with proper spelling and grammar, only rerurn the corrected sentence.
# Please act as a seq2seq model and correct the typos in the following sentence. Respond according to the input language.
# 請用正確的拼寫和語法修正以下文字。
# この文の誤字を修正してください。修正した正しい単語のみを返してください。誤りがなければ元の文をそのまま返してください。

# この文にはスペルミスがあります
# 今日は本当に天気がいいです
# 本お読む
# 学校え行く
# 友達え話す
# 你說你想要逃偏偏注定要洛角
# What flights are available from New York to Los Angeles today?

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
    extracted_texts = re.findall(r'【(.*?)】', formatted_text)  # 半形引號 "
    result = "\n".join(extracted_texts)
    # with open("response.txt", "w", encoding="utf-8") as file:
    #     file.write(result)
    end_time = time.time()
    print("\n")
    print(f"Execution time: {end_time - start_time} seconds")
else:
    print("Error:", response.status_code, response.text)