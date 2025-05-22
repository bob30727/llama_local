import requests
import json
import time
import re

# 設定 Ollama API 端點與模型名稱
OLLAMA_API_URL = "http://localhost:11434/api/chat"
MODEL_NAME = "llama3.1"

# 讀取輸入的 TXT 檔案
INPUT_FILE = "input.txt"
OUTPUT_FILE = "output.jsonl"

# 讀取所有句子
with open(INPUT_FILE, "r", encoding="utf-8") as f:
    sentences = [line.strip() for line in f.readlines() if line.strip()]

# 設定 LLaMA3 提示詞
SYSTEM_PROMPT = """
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

Enclose the identified category in 【】. Output only the category.
"""

# 定義函式來呼叫 LLaMA3
def query_llama3(sentence):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"the sentence is : {sentence}"}
    ]

    response = requests.post(
        OLLAMA_API_URL,
        json={"model": MODEL_NAME, "messages": messages, "temperature": 0},
        stream=True
    )

    # 解析回應
    if response.status_code == 200:
        contents = []
        for line in response.iter_lines(decode_unicode=True):
            if line.strip():
                try:
                    data = json.loads(line)  # 解析 JSON
                    content = data.get("message", {}).get("content", "")
                    if content:
                        contents.append(content)
                except json.JSONDecodeError:
                    continue

        formatted_text = "".join(contents)
        match = re.search(r'【(.*?)】', formatted_text)  # 提取【】中的內容
        return match.group(1) if match else "Unknown"
    else:
        print(f"Error: {response.status_code}")
        return "Unknown"

# 逐句處理並寫入 JSONL 檔案
with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
    for sentence in sentences:
        label = query_llama3(sentence)
        json_entry = {"text": sentence, "label": label}
        f.write(json.dumps(json_entry, ensure_ascii=False) + "\n")

print(f"已處理完成，結果儲存於 {OUTPUT_FILE}")