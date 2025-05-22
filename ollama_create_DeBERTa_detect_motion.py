import requests
import json
import time
import re

# 設定 Ollama API 端點與模型名稱
OLLAMA_API_URL = "http://localhost:11434/api/chat"
MODEL_NAME = "llama3.1"
OUTPUT_FILE = "output_motion.jsonl"

file_path = "output_2.jsonl"  # 修改為你的 jsonl 檔案路徑
data = []

gestures_map = {
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

# 讀取 JSONL 檔案
with open(file_path, "r", encoding="utf-8") as f:
    for line in f:
        data.append(json.loads(line.strip()))




# 定義函式來呼叫 LLaMA3
def query_llama3(sentence, SYSTEM_PROMPT):
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
        print(formatted_text)
        print("+++++++++++++++")
        match = re.search(r'【(.*?)】', formatted_text)  # 提取【】中的內容
        return match.group(1) if match else "Unknown1"
    else:
        print(f"Error: {response.status_code}")
        return "Unknown2"

with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
    for item in data:
        text = item["text"]
        label = item["label"]
        if label in gestures_map:
            gestures = "\n".join(f"- {gesture}" for gesture in gestures_map[label])
            prompt = f'The following sentence is most likely associated with {label}\nWhich corresponding body gesture does it match：\n{gestures}\nEnclose the identified body gesture in 【】. Output only the category.such as:【Nodding】\n【Shaking head】\n【Extending hand for a handshake】'
            motion_label = query_llama3(text, prompt)
            print(text)
            print(motion_label)
            print("=======================================")
            if label == "Unknown":
                print(f"❌ 跳過: {text}（分類失敗）")
                continue  # 跳過儲存這一條
            json_entry = {"text": text, "label": motion_label}
            f.write(json.dumps(json_entry, ensure_ascii=False) + "\n")
            f.flush()  # 立即寫入磁碟，防止中途崩潰時資料遺失
            time.sleep(0.5)  # 可調整間隔時間，避免過度請求 API

