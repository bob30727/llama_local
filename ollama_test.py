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
Task:
Classify the given sentence into one of the six gesture categories based on its meaning and emotional intent. Then, suggest an appropriate body gesture that would naturally accompany this sentence.
Gesture Categories:

Positive Expressions (Usually associated with approval, joy, or encouragement)
Example Sentences & Gestures:
“That’s a great idea!” (Nodding, smiling)
“Awesome! We did it!” (Arms open wide, excitedly waving hands)
“You’re right, I completely agree.” (Nodding, leaning forward)
“Congratulations! This is a moment worth celebrating.” (Reaching out, possibly accompanied by a hug)

Negative Expressions (Associated with disagreement, dissatisfaction, or disappointment)
Example Sentences & Gestures:
“I don’t think this is a good idea...” (Shaking head, frowning)
“Sigh, how did things end up like this?” (Arms crossed, looking down, sighing)
“This really bothers me.” (Frowning, leaning back)
“Are you sure this is the right thing to do?” (Arms crossed, looking at the person skeptically)

Emphasizing Gestures (Used to reinforce tone or highlight key points)
Example Sentences & Gestures:
“This is important, you must remember it!” (Pointing at the person or the table)
“Listen to me, what I mean is...” (Waving hands to emphasize speech)
“We need to focus on this issue!” (Firmly slapping the table)
“This is no joke.” (Hands open wide, showing a serious expression)

Interactive Gestures (Used to facilitate conversation or build connections)
Example Sentences & Gestures:
“Hey! Long time no see!” (Extending hand for a handshake or opening arms for a hug)
“Excuse me, may I interrupt?” (Leaning slightly forward, gesturing with hand)
“Let me help you!” (Reaching out to assist, moving closer to the person)
“What do you think about this?” (Making eye contact, turning body towards the person)

Thinking Gestures (Related to thinking, hesitation, or uncertainty)
Example Sentences & Gestures:
“This is a bit tricky...” (Touching chin or frowning in thought)
“Let me think, hmm...” (Tapping fingers on forehead or scratching head)
“I feel like I’ve seen this somewhere before...” (Looking down, deep in thought)
“You mean...?” (Tilting head slightly to show confusion)

Anxious/Defensive Gestures (Associated with nervousness, stress, or defensive behavior)
Example Sentences & Gestures:
“Uh... I’m not too sure...” (Fidgeting with fingers or an object, avoiding eye contact)
“I feel a little uncomfortable...” (Hugging oneself, slightly stepping back)
“This really makes me anxious...” (Shuffling feet, clenching fists)
“Can we change the topic?” (Looking down, hands crossed in front)

Enclose the identified category in 【】. output categorie only.
such as:
【Positive Expressions】
【Negative Expressions】
【Emphasizing Gestures】
【Interactive Gestures】
【Thinking Gestures】
【Anxious/Defensive Gestures】

"""},
    {"role": "user", "content": "the sentence is : Do you know where the nearest bus stop is?"}
]

# output as JSONL format, like:
# {"text": "I love this product! It's amazing.", "label": "positive"}
# {"text": "This is the worst experience ever.", "label": "negative"}
# {"text": "The movie was okay, not great but not bad either.", "label": "neutral"}

# Enclose the identified category in 【】. output the probabilities of these six categories.
# such as:
# 【Positive Expressions】70%
# 【Negative Expressions】20%
# 【Emphasizing Gestures】10%
# 【Interactive Gestures】0%
# 【Thinking Gestures】0%
# 【Anxious/Defensive Gestures】0%

# Task: Identify the category of the given sentence based on the provided classifications. Enclose the identified category in 【】. Output category only.
# Categories:
# Task Management - Involves exchanging information related to the main task of the conversation, such as asking questions, giving advice, or responding to task-related queries.
# Turn Management - Refers to regulating the flow of conversation, such as requesting to speak, yielding the floor, interrupting, or signaling the end of a turn.
# Time Management - Pertains to managing time-related aspects of a conversation, like scheduling meetings, asking about availability, or reminding about time.
# Discourse Structuring - Organizing and shaping the conversation, such as introducing new topics, transitioning between ideas, or summarizing discussions.
# Own Communication Management - Managing one’s own communication, such as correcting mistakes, rephrasing statements, or clarifying meaning.
# Partner Communication Management - Influencing the way the conversation partner communicates, such as asking them to speak slower, louder, or clarify their statements.
# Auto-Feedback - Reflects the speaker’s understanding of their own output, like confirming whether their statement is clear or acknowledging uncertainty.
# Allo-Feedback - Reflects the understanding of the other participant in the conversation, such as asking for repetition or confirming comprehension.
# Social Obligations Management - Relates to managing social niceties in the conversation, like greetings, apologies, expressing gratitude, or making polite requests.

# Output category only.
# The previous sentence is ''

test_sentences = [
    "Hey! How are you today?",  # Social Obligations Management
    "Do you know where the nearest bus stop is?",  #Task Management
    "Let's go grab some coffee later.",  #Task Management
    "No, I don’t want to go out today.",  #Own Communication Management
    "I'm sorry, I can't make it to the meeting.",  #Social Obligations Management
    "It is 3 PM now.", #Time Management
    "I think this movie is great.", #Auto-Feedback
    "Do you like coffee?", #Task Management
    "Do you want this?", #Task Management
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
    extracted_texts = re.findall(r'【(.*?)】', formatted_text)  # 半形引號 "
    result = "\n".join(extracted_texts)
    with open("response.txt", "w", encoding="utf-8") as file:
        file.write(result)

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