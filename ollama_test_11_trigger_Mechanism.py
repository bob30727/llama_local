import requests
import json
import time
import re

# 設定 Ollama API 端點
OLLAMA_API_URL = "http://localhost:11434/api/chat"
# OLLAMA_API_URL = "http://ollama_server:11434/api/chat"

# 指定模型名稱
MODEL_NAME = "llama3.1"

motion_counter = {}
motion_cooldown = {}
recent_motions = []

# **主函式**
def main(sentence: str):
    formatted_sentence = "the sentence is : " + sentence

    # 取得 LLM 回應
    response_text = get_llm_response(formatted_sentence)
    print(response_text)
    print("================")

    if response_text:
        start_time = time.time()
        # 解析回應並格式化 JSON
        output_json = parse_response(response_text)

        # 輸出 JSON 結果
        print(json.dumps(output_json, ensure_ascii=False, indent=2))
        json_string = json.dumps(output_json, ensure_ascii=False, indent=2)
        end_time = time.time()

        # with open("ollama_output_3.txt", "a", encoding="utf-8") as file:
        #     file.write(f"Processing sentence: {sentence}\n")
        #     file.write(json_string)
        #     file.write(f"\nExecution time: {end_time - start_time:.2f} seconds\n")
        #     file.write("==================================================================\n")


# **發送請求至 LLM**
def get_llm_response(sentence: str) -> str:
    messages = [
        {"role": "system", "content": """
Act as a seq2seq model and output text according to the following rules and requirements.  

Step 1: Identify which part of this sentence is most suitable for a physical movement and use [@timestamp_begin] to mark the keyword where the physical movement should be made. These keywords could include verbs or adjectives that are directly associated with the primary action or purpose of the sentencen, but each sentences will have only one [@timestamp_begin].
Verbs typically describe actions or activities, serving as the core of the action. Verbs example: 
"I'd be happy to [@timestamp_begin]introduce you to our latest mobile phone models."
"we have model A, which [@timestamp_begin]packs a powerful processor and plenty of storage space for all your apps and files."
"Its advanced battery life also [@timestamp_begin]means you can use it all day without needing to recharge." 
"If you're looking for something with more power, I [@timestamp_begin]recommend this high-performance model."

Adjectives are used to describe the characteristics, state, or quality of nouns, and they help to strengthen the tone of the sentence. Adjectives example:
"This is an [@timestamp_begin]incredible opportunity for us."
"That sounds [@timestamp_begin]amazing, I’m so excited for you!"
"I understand how [@timestamp_begin]frustrating that must be."

No need for quotation marks. If there are multiple sentences, output one result per sentence. 
Output format:  
`###STEP1### <sentence 1>`   

Step 2: Determine whether the meaning of the sentences is positive, negative, or neutral.
Output format:  
`###STEP2### <positive/negative/neutral>`

Step 3: Select the most appropriate body gesture from the list below based on the context and meaning of these sentences.
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

only output title, such as : Greeting Group, or Product Showcase. If there are multiple sentences, output one result per sentence.  
Output format:  
`###STEP3### <selected body gesture>`  

### Example Input:
"Nice to meet you. I'd be happy to help you explore our latest products! Can I show you something specific today?"

### Expected Output:
###STEP1### Nice to meet you. I'd be happy to help you [@timestamp_begin]explore our latest products! Can I show you something specific today?

###STEP2### positive

###STEP3### Product Showcase

"""},
        {"role": "user", "content": sentence}
    ]

    try:
        response = requests.post(
            OLLAMA_API_URL,
            json={"model": MODEL_NAME, "messages": messages, "temperature": 0},
            timeout=10,  # 設定超時時間，避免請求卡住
            stream=True  # 啟用流式回應
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
    global motion_counter, recent_motions

    texts = extract_all_with_regex(r'###STEP1###\s*(.*)', response_text)
    intentions = extract_all_with_regex(r'###STEP2###\s*(.*?)\s*(?=###STEP2###|$)', response_text)
    motions = extract_all_with_regex(r'###STEP3###\s*(.*?)\s*(?=###STEP3###|$)', response_text)
    # r'###STEP3###\s*([\s\S]*?)(?=###STEP3###|$)'

    # 整合文字內容
    combined_text = " ".join(texts)
    overall_intention = determine_overall_intention(intentions)

    valid_motions = []

    # **處理冷卻機制**
    motions_to_remove = []
    for motion in motion_cooldown:
        motion_cooldown[motion] -= 1  # 冷卻時間 -1
        if motion_cooldown[motion] <= 0:
            motions_to_remove.append(motion)  # 冷卻完成，移除該動作

    for motion in motions_to_remove:
        del motion_cooldown[motion]
        if motion in recent_motions:
            recent_motions.remove(motion)  # 從 recent_motions 刪除，讓它可以再次觸發

    # **處理當前輸入的動作**
    for motion in motions:
        if motion not in motion_counter:
            motion_counter[motion] = 1
        else:
            motion_counter[motion] += 1

        # **當某動作計數 >= 2，且不在 recent_motions，則觸發**
        if motion_counter[motion] >= 2 and motion not in recent_motions:
            valid_motions.append(motion)
            recent_motions.append(motion)  # 記錄最近觸發的動作
            if motion == "Product Showcase":
                motion_cooldown[motion] = 2
            elif motion == "Greeting Group":
                motion_cooldown[motion] = 6
            else:
                motion_cooldown[motion] = 3
            motion_counter[motion] = 0  # 觸發後重置計數

    return {
        "text": combined_text,
        "intention": overall_intention,
        "motion_tag": [{"motion": motion} for motion in valid_motions]
    }


# **執行程式**
if __name__ == "__main__":
    while True:
        test_sentences = input("Question : ")
        main(test_sentences)
