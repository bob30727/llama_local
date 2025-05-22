import requests
import json
import time
import re

# 設定 Ollama API 端點
OLLAMA_API_URL = "http://localhost:11434/api/chat"

# 指定模型名稱
MODEL_NAME = "llama3.2:1b"

# **主函式**
def main(sentence: str):
    start_time = time.time()
    sentence = "the sentence that tou need to detect is : " + sentence

    # 取得 LLM 回應
    response_text = get_llm_response(sentence)
    print(response_text)
    print("=========================================")

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
Act as a seq2seq model and output text according to the following rules and requirements.  

Step 1: using [@timestamp_begin] mark the keywords in each sentence that are related to actions or the intent behind the sentence. These keywords could include verbs or adjectives that are directly associated with the primary action or purpose of the sentencen, but each sentence will have only one [@timestamp_begin].
Verbs typically describe actions or activities, serving as the core of the action. Verbs example: 
"I'd be happy to [@timestamp_begin]introduce you to our latest mobile phone models."
"we have model A, which [@timestamp_begin]packs a powerful processor and plenty of storage space for all your apps and files."
"Its advanced battery life also [@timestamp_begin]means you can use it all day without needing to recharge." 
"If you're looking for something with more power, I [@timestamp_begin]recommend this high-performance model."

Adjectives are used to describe the characteristics, state, or quality of nouns, and they help to strengthen the tone of the sentence. Adjectives example:
"This is an [@timestamp_begin]incredible opportunity for us."
"That sounds [@timestamp_begin]amazing, I’m so excited for you!"
"I understand how [@timestamp_begin]frustrating that must be."

No need for quotation marks.
Output format:  
`###STEP1### <sentence 1>`  

Step 2: Determine whether the meaning of the sentence is positive, negative, or neutral. If there are multiple sentences, output one result per sentence. 
Output format:  
`###STEP2### <positive/negative/neutral>`

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

only output title, such as : Greeting Group, or Product Showcase. 
Output format:  
`###STEP3### <selected body gesture>`  

### Example Input:
"I'd be happy to help you explore our latest products!"

### Expected Output:
###STEP1### I'd be happy to [@timestamp_begin]help you explore our latest products! 

###STEP2### positive

###STEP3### Product Showcase

"""},
        {"role": "user", "content": sentence}
    ]

    try:
        response = requests.post(
            OLLAMA_API_URL,
            json = {"model": MODEL_NAME, "messages": messages, "temperature": 0},
            timeout = 10,  # 設定超時時間，避免請求卡住
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

def extract_all_with_regex(pattern: str, text: str):
    """ 使用正則表達式抓取所有符合條件的值，並回傳列表 """
    return re.findall(pattern, text)

def determine_overall_intention(intention_list):
    """ 判斷整體語句的情緒，如果有 positive，則結果為 positive；否則選擇最強烈的情緒 """
    if "positive" in intention_list:
        return "positive"
    elif "negative" in intention_list:
        return "negative"
    return "neutral"

# **解析 LLM 回應**
def parse_response(response_text: str) -> dict:
    """
    使用正則表達式解析 LLM 回應，擷取動詞標記的文本、情緒、適當的肢體語言。
    """
    # text = extract_with_regex(r'###STEP1###\s*(.*?)\s*(?=###STEP2###)', response_text)
    texts = extract_all_with_regex(r'###STEP1###\s*(.*)', response_text)

    intentions = extract_all_with_regex(r'###STEP2###\s*(.*?)\s*(?=###STEP2###|$)', response_text)

    motions = extract_all_with_regex(r'###STEP3###\s*(.*?)\s*(?=###STEP3###|$)', response_text)

    # 整合文字內容
    combined_text = " ".join(texts)

    overall_intention = determine_overall_intention(intentions)

    # 建立 motion_tag 清單
    # motion_tags = [{"ID": idx + 1, "motion": motion} for idx, motion in enumerate(motions)]
    motion_tags = [{"ID": idx + 1, "motion": motion} for idx, (motion, _) in enumerate(zip(motions, texts))]

    return {
        "text": combined_text,
        "intention": overall_intention,
        "motion_tag": motion_tags
    }

# **執行程式**
if __name__ == "__main__":
    test_sentence = "I’m sorry, but it seems there’s a system issue at the moment."
    main(test_sentence)