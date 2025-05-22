import requests
import json
import time
import re

# 設定 Ollama API 端點
OLLAMA_API_URL = "http://localhost:11434/api/chat"

# 指定模型名稱
MODEL_NAME = "llama3.1"


# **主函式**
def main(sentence: str):
    start_time = time.time()
    sentence = "the sentence is : " + sentence

    # 取得 LLM 回應
    response_text = get_llm_response(sentence)

    if response_text:
        # 解析回應並格式化 JSON
        output_json = parse_response(response_text)

        # 輸出 JSON 結果
        print(json.dumps(output_json, ensure_ascii=False, indent=2))

    end_time = time.time()
    print(f"\nExecution time: {end_time - start_time:.2f} seconds")


# **發送請求至 LLM**
def get_llm_response(sentence: str) -> str:
    messages = [
        {"role": "system", "content": """
Analyze the given sentence, infer the speaker's intent, and determine the most appropriate body language.  

Step 1: Mark verbs in the sentence using [@timestamp_begin].  
For example: 
"I'd be happy to [@timestamp_begin]introduce you to our latest mobile phone models."
"we have model A, which [@timestamp_begin]packs a powerful processor and plenty of storage space for all your apps and files."
"Its advanced battery life also [@timestamp_begin]means you can use it all day without needing to recharge." 
"Next up is model B, which [@timestamp_begin]offers a unique blend of style and functionality."
"If you're looking for something with more power, I [@timestamp_begin]recommend this high-performance model."
No need for quotation marks. Output format: `###STEP1### <modified sentence>`

Step 2: Determine whether the action in the sentence is positive, negative, or neutral.  
Output format: `###STEP2### <positive/negative/neutral>`

Step 3: Select the most appropriate body gesture from the list below based on the context and meaning of this sentence.

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

only output title, such as : Greeting Group, or Product Showcase. Output format: `###STEP3### <selected body gesture>`
"""},
        {"role": "user", "content": sentence}
    ]

    try:
        response = requests.post(
            OLLAMA_API_URL,
            json={"model": MODEL_NAME, "messages": messages, "temperature": 0},
            timeout=10,  # 設定超時時間，避免請求卡住
            stream = True  # 啟用流式回應
        )

        response.raise_for_status()  # 若 API 回應錯誤，拋出例外
        response_text = ""
        for line in response.iter_lines(decode_unicode=True):
            if line.strip():  # 確保每行有內容
                try:
                    data = json.loads(line)  # 解析單行 JSON
                    message_content = data.get("message", {}).get("content", "")
                    if message_content:
                        response_text += message_content  # 合併內容
                except json.JSONDecodeError as e:
                    print(f"JSON 解析錯誤: {e}")

        return response_text.strip()

    except requests.exceptions.RequestException as e:
        print(f"API request failed: {e}")
        return ""


# **解析 LLM 回應**
def parse_response(response_text: str) -> dict:
    """
    使用正則表達式解析 LLM 回應，擷取動詞標記的文本、情緒、適當的肢體語言。
    """
    text = extract_with_regex(r'###STEP1###\s*(.*?)\s*(?=###STEP2###)', response_text)
    intention = extract_with_regex(r'###STEP2###\s*(.*?)\s*(?=###STEP3###|###STEP4###|$)', response_text,
                                   default="neutral")
    motion = extract_with_regex(r'###STEP3###\s*(.*)', response_text, default="None")

    return {
        "text": text,
        "intention": intention,
        "motion_tag": [{"ID": 1, "motion": motion}]
    }


# **使用正則表達式擷取資料**
def extract_with_regex(pattern: str, text: str, default: str = "") -> str:
    """
    用正則表達式擷取文字，若找不到則回傳預設值。
    """
    match = re.search(pattern, text, re.DOTALL)
    return match.group(1).strip() if match else default

# I’d be happy to help you explore our latest products!
# Can I show you something specific today?"
# **執行程式**
if __name__ == "__main__":
    test_sentence = "I’m sorry, but it seems there’s a system issue at the moment."
    main(test_sentence)