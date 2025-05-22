import requests
import json
import time
import re

# 設定 Ollama API 的端點
OLLAMA_API_URL = "http://localhost:11434/api/chat"

# 指定模型名稱
# model_name = "llama3.2:1b"
model_name = "llama3.1"

sentence = "If you have any questions, feel free to ask."
sentence = "the sentence is : " + sentence

messages_1 = [
    {"role": "system", "content": """
Task:
using [@timestamp_begin] mark the verbs in each sentence.

For example:
"I'd be happy to [@timestamp_begin]introduce you to our latest mobile phone models."
"we have model A, which [@timestamp_begin]packs a powerful processor and plenty of storage space for all your apps and files."
"Its advanced battery life also [@timestamp_begin]means you can use it all day without needing to recharge."
"Next up is model B, which [@timestamp_begin]offers a unique blend of style and functionality."
"If you're looking for something with more power, I [@timestamp_begin]recommend this high-performance model."

Only output the modified sentence.
"""},
    {"role": "user", "content": sentence}
]


messages_2 = [
    {"role": "system", "content": """
Task:
Select the most appropriate body gesture from the list below based on the context and meaning of this sentence.

- Greeting Group : Actions used to welcome a user when they approach.
Example sentences:
"Hello! How can I help you today?"
"Welcome to our store, we're glad to have you here!"
"Hi there, nice to meet you!"

- Farewell Set : Actions used to bid farewell at the end of an interaction.
Example sentences:
"Thanks for visiting, take care!"
"Goodbye! Have a great day ahead!"
"It was a pleasure helping you, see you next time!"

- Emotion Set : Actions that express the user's emotions.
Example sentences:
"I’m so happy to see you!"
"I understand how frustrating that must be."
"That sounds amazing, I’m so excited for you!"

- Idle Animations : Actions displayed during periods of no user interaction.
Example sentences:
"The system is waiting for your input."
"Please hold on, we're processing your request."
"We're just getting things ready for you."

- Product Showcase : Actions used to showcase or display a product.
Example sentences:
"Let me show you our latest smartphone, featuring a 48MP camera."
"This is our newest model, designed for long battery life."
"Take a look at this sleek laptop with advanced features."

- Navigation : Actions performed when moving from one location to another.
Example sentences:
"Let me take you to the checkout page."
"I’ll guide you to the product you’re looking for."
"Click here to navigate to your account settings."

- Error Handling : Actions shown when the system encounters an error.
Example sentences:
"Sorry, something went wrong. Please try again."
"Oops! It looks like we encountered an error."
"There was an issue processing your request, please check again."

- Listening State : Actions indicating the system is listening to the user's input.
Example sentences:
"I’m listening, please go ahead."
"I’m waiting for your response."
"Please tell me more, I’m ready to listen."

- Talking State : Gestures that accompany the system's spoken response.
Example sentences:
"I’m explaining how it works."
"Here’s what you need to know about this product."
"Let me walk you through the process."

only output title, such as : Greeting Group, or Product Showcase.

"""},
    {"role": "user", "content": sentence}
]


messages_3 = [
    {"role": "system", "content": """
Task: 
Determine whether the meaning of the sentence is positive, negative, or neutral. 
only output title, such as : positive
"""},
    {"role": "user", "content": sentence}
]

start_time = time.time()
# 發送請求給 Ollama
# response1 = requests.post(
#     OLLAMA_API_URL,
#     json={
#         "model": model_name,
#         "messages": messages_1,
#         "temperature" : 0
#     },
#     stream=True  # 啟用流式處理
# )
# response2 = requests.post(
#     OLLAMA_API_URL,
#     json={
#         "model": model_name,
#         "messages": messages_2,
#         "temperature" : 0
#     },
#     stream=True  # 啟用流式處理
# )
response3 = requests.post(
    OLLAMA_API_URL,
    json={
        "model": model_name,
        "messages": messages_3,
        "temperature" : 0
    },
    stream=True  # 啟用流式處理
)

# 檢查回應狀態碼
if response3.status_code == 200:
    print("回覆:")
    # 處理返回的多行 JSON

    # contents_1 = []  # 用來存儲所有內容
    # for line in response1.iter_lines(decode_unicode=True):
    #     if line.strip():  # 確保每行有內容
    #         try:
    #             data = json.loads(line)  # 將每行 JSON 解析為字典
    #             content = data.get("message", {}).get("content", "")
    #             if content:
    #                 print(content, end="")  # 按順序顯示回應內容
    #                 contents_1.append(content) # 存到列表
    #         except json.JSONDecodeError as e:
    #             print(f"\nJSON 解碼錯誤: {e}")
    # print("\n")
    # formatted_text = "".join(contents_1)

    # contents_2 = []  # 用來存儲所有內容
    # for line in response2.iter_lines(decode_unicode=True):
    #     if line.strip():  # 確保每行有內容
    #         try:
    #             data = json.loads(line)  # 將每行 JSON 解析為字典
    #             content = data.get("message", {}).get("content", "")
    #             if content:
    #                 print(content, end="")  # 按順序顯示回應內容
    #                 contents_2.append(content)  # 存到列表
    #         except json.JSONDecodeError as e:
    #             print(f"\nJSON 解碼錯誤: {e}")
    # print("\n")
    # formatted_text = "".join(contents_2)

    contents_3 = []  # 用來存儲所有內容
    for line in response3.iter_lines(decode_unicode=True):
        if line.strip():  # 確保每行有內容
            try:
                data = json.loads(line)  # 將每行 JSON 解析為字典
                content = data.get("message", {}).get("content", "")
                if content:
                    print(content, end="")  # 按順序顯示回應內容
                    contents_3.append(content)  # 存到列表
            except json.JSONDecodeError as e:
                print(f"\nJSON 解碼錯誤: {e}")
    formatted_text = "".join(contents_3)


    end_time = time.time()
    print("\n")
    print(f"Execution time: {end_time - start_time} seconds")
else:
    print("Error:", response.status_code, response.text)