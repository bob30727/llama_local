import requests
import json
import time
import re
import os

# 設定 Ollama API 端點與模型名稱
OLLAMA_API_URL = "http://localhost:11434/api/chat"
MODEL_NAME = "llama3.1"

# 設定 JSONL 輸出檔案
OUTPUT_FILE = "output_motion_2.jsonl"

# 定義手勢分類對應表
GESTURES_MAP = {
    "Positive Expressions": [
        "Nodding", "Smiling", "Arms open wide",
        "Excitedly waving hands", "Leaning forward", "Reaching out"
    ],
    "Negative Expressions": [
        "Shaking head", "Frowning", "Arms crossed",
        "Looking down", "Sighing", "Leaning back",
        "Looking skeptically at the person"
    ],
    "Emphasizing Gestures": [
        "Pointing at the person or the table", "Waving hands to emphasize speech",
        "Firmly slapping the table", "Hands open wide"
    ],
    "Interactive Gestures": [
        "Extending hand for a handshake", "Opening arms for a hug",
        "Leaning slightly forward", "Gesturing with hand",
        "Reaching out to assist", "Making eye contact",
        "Turning body towards the person"
    ],
    "Thinking Gestures": [
        "Touching chin", "Frowning in thought",
        "Tapping fingers on forehead", "Scratching head",
        "Looking down, deep in thought", "Tilting head slightly to show confusion"
    ],
    "Anxious/Defensive Gestures": [
        "Fidgeting with fingers or an object", "Avoiding eye contact",
        "Hugging oneself", "Slightly stepping back",
        "Hands crossed in front", "Shuffling feet",
        "Clenching fists"
    ]
}

# 系統提示詞：分類句子為肢體動作類別
SYSTEM_PROMPT_INTENTION = """
Task:
Classify the given sentence into one of the seven gesture categories based on its meaning and emotional intent. Then, suggest an appropriate body gesture that would naturally accompany this sentence.
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

No Gesture Needed (Some sentences are purely informational and usually do not require specific gestures.)
Example Sentences & Gestures:
"It is 3 PM now." (No gesture needed)
"Please order a cup of coffee for me." (No gesture needed)
"I like drinking tea." (No gesture needed)
"Pass me a piece of paper." (No gesture needed)

Enclose the identified category in 【】. Output only the category.
"""

# 呼叫 LLaMA3 API 的函式
def query_llama3(sentence, system_prompt):
    """向 LLaMA3 查詢並解析 API 回應"""
    try:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"the sentence is: {sentence}"}
        ]

        response = requests.post(
            OLLAMA_API_URL,
            json={"model": MODEL_NAME, "messages": messages, "temperature": 0},
            timeout=10  # 設定超時
        )

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
            match = re.search(r'【(.*?)】', formatted_text)
            return match.group(1) if match else "Unknown"
        else:
            print(f"API Error: {response.status_code}")
            return "Unknown"

    except requests.exceptions.RequestException as e:
        print(f"API Request Failed: {e}")
        return "Unknown"

# 讀取文本檔案
def read_sentences_from_file(file_path):
    """從文本檔案讀取每一行作為句子"""
    if not os.path.exists(file_path):
        print(f"Error: File '{file_path}' not found!")
        return []

    with open(file_path, "r", encoding="utf-8") as file:
        sentences = [line.strip() for line in file if line.strip()]
    return sentences

# 主要處理邏輯
def process_sentences(file_path):
    """處理文本句子，並輸出 JSONL 格式"""
    sentences = read_sentences_from_file(file_path)
    results = []

    for sentence in sentences:
        print(f"Processing: '{sentence}'")

        # 判斷句子對應的肢體動作類別
        intension_label = query_llama3(sentence, SYSTEM_PROMPT_INTENTION)
        print(f"  → 動作類別: {intension_label}")

        if intension_label.lower() == "unknown":
            print("  → 不用做動作")
            label = "Unknown"
        else:
            gestures = "\n".join(f"- {gesture}" for gesture in GESTURES_MAP.get(intension_label, []))
            system_prompt_motion = f'The following sentence is most likely associated with {intension_label}\nWhich corresponding body gesture does it match:\n{gestures}\nEnclose the identified body gesture in 【】. Output only the category.'

            # 取得具體的肢體動作
            motion_label = query_llama3(sentence, system_prompt_motion)
            print(f"  → 該做出的動作: {motion_label}")
            label = motion_label if motion_label != "Unknown" else intension_label

        # 儲存結果
        results.append({"text": sentence, "label": label})

        # 輸出 JSONL 檔案
        with open(OUTPUT_FILE, "a", encoding="utf-8") as out_file:
            for item in results:
                json.dump(item, out_file, ensure_ascii=False)
                out_file.write("\n")
                out_file.flush()
        results = []



    print(f"✅ 處理完成，輸出文件: {OUTPUT_FILE}")

# 執行主函式
if __name__ == "__main__":
    input_file = "output_qna_lines.txt"  # 替換成你的 txt 檔案名稱
    process_sentences(input_file)