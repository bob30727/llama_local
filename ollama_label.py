import requests
import json
import time
import re

# 設定 Ollama API 的端點
OLLAMA_API_URL = "http://localhost:11434/api/chat"

# 指定模型名稱
# model_name = "llama3.2:1b"
model_name = "llama3.1"

with open("response_1.txt", "r", encoding="utf-8") as file:
    queries = [line.strip() for line in file if line.strip()]  # 去除空行

# 設定系統提示詞
system_message = """
1. Flight_info: 
is General flight information queries and flight number
Ex: “What flights are available from New York to Los Angeles?”
Ex: “Show me flights from Dallas to Chicago on Monday.”

2. Airfare:  
is Queries about ticket prices
Ex:  “How much is a first-class ticket from San Francisco to Boston?”
Ex:“What’s the cheapest fare from Seattle to Miami?”

3. Airline: 
is Questions related to specific airlines and their services.
Ex: “Which airlines fly from Boston to Seattle?”
Ex: “Is Delta Airlines operating flights from Los Angeles to Chicago?”
     
4. Flight_Status: 
is Real-time updates or delays,arrival, departure ,and cancellations
Ex: “What time does the flight from New York to Miami depart?”
Ex: “When does the next flight from Denver to Las Vegas arrive?”
Ex: “Is my flight on time?”
Ex: “Has the flight from JFK to LAX been delayed?”
Ex: “What’s the current status of Flight AA127?”

5. Flight_meal:  
is Meal Availability on Flights
Ex: “Does United Airlines provide vegetarian meals on flights from New York to London?”
Ex: “Are meals included in the economy class ticket for flights to Paris?”

6. class_seat_inquary: 
is Seat Availability Queries(atis_class_type + atis_capacity)
Ex: “Are any seats available on the 6 PM flight from Los Angeles to Denver?”
Ex: “How many seats are left on the next flight to Orlando?”
    
7. Booking and Reservation:  
is Queries about flight booking, reservations, modifications, or cancellations.
Ex: “How can I book a flight from New York to London?”
Ex: “Can I change my flight date after booking?”
Ex: “What is the cancellation policy for my reservation?”
	
8. Weather:  
is Queries about the weather at departure, arrival locations, or potential weather-related flight delays.
Ex: “What’s the weather like in Paris today?”
Ex: “Is my flight delayed due to the storm?”
Ex: “How cold is it in Chicago right now?”

9. Luggage Handling: 
is Questions related to baggage check-in, baggage fees, and restrictions.
Ex: “What is the baggage allowance for my flight?”
Ex: “How do I track my  luggage?”
Ex: “Are thereExtra fees for oversized baggage?”

10. Security or Regulation Query: 
is Queries related to security procedures, travel restrictions, and airline policies.
Ex: “What are the security screening requirements for international flights?”
Ex: “Can I bring a laptop in my carry-on bag?”
Ex: “Are there any travel restrictions due to COVID-19?”

11. Lost and Found: 
is Queries related to lost items, misplaced baggage, or retrieving forgotten belongings at the airport or on a flight.
Ex: “I left my phone on the plane. How can I get it back?”
Ex: “Where can I report a lost item at the airport?”
Ex: “Is there a lost and found department at JFK Airport?”
	
12. ground_fare: 
is Questions about fares for ground transportation services like taxis, shuttles, or buses, Ground Transport Fare
Ex:“How much is a taxi from JFK Airport to Times Square?”
Ex: “What is the fare for the airport shuttle to downtown Los Angeles?

13. Airport Facilities:   
is Queries about airport amenities, facilities available for passenger services, Inquiries about VIP lounges, premium services, or additional travel benefits. important for passengers needingExtra assistance (elderly, disabled, unaccompanied minors).
Ex: “Does the airport have free Wi-Fi?”
Ex: “Where can I find a charging station at the airport?”
Ex: “Are there any restaurants open late at the airport?”
Ex: “Is there a kids’ play area in the terminal?”
Ex: “Where can I find an ATM at the airport?”
Ex: “How can I book access to the VIP lounge?”
Ex: “Does my first-class ticket include a private waiting area?”
Ex: “Are there anyExclusive airport services for business travelers?”
Ex: “Can I request wheelchair assistance at the airport?”
Ex: “Do you provide special assistance for elderly travelers?”

14. Pet_Travel_&_Animal Transport: 
is Pet-friendly airline policies
Ex: “Can I bring my small dog in the cabin with me?”
Ex: “What are the airline rules for traveling with pets?”
Ex: “Is there a pet-friendly flight option?”

You are a classifier. 
Please refer to the examples of the 14 types of intentions above and help determine which type of intention the following sentence belongs to.
The selected intention enclosed in specific symbols 【】.

for example:
【Flight_info】
【Airfare】
【Airline】
【Flight_Status】
【Flight_meal】
【class_seat_inquary】
【Booking and Reservation】
【Weather】
【Luggage Handling】
【Security or Regulation Query】
【Lost and Found】
【ground_fare】
【Airport Facilities】
【Pet_Travel_&_Animal Transport】

"""

for query in queries:
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": query}
    ]
    # 發送請求給 Ollama
    response = requests.post(
        OLLAMA_API_URL,
        json={
            "model": model_name,
            "messages": messages,
            "temperature": 0
        },
        stream=True  # 啟用流式處理
    )
    # 檢查回應狀態碼
    if response.status_code == 200:
        contents = []  # 用來存儲所有內容
        output_lines = []
        for line in response.iter_lines(decode_unicode=True):
            if line.strip():  # 確保每行有內容
                try:
                    data = json.loads(line)  # 將每行 JSON 解析為字典
                    content = data.get("message", {}).get("content", "")
                    if content:
                        contents.append(content)  # 存到列表
                except json.JSONDecodeError as e:
                    print(f"\nJSON 解碼錯誤: {e}")
        formatted_text = "".join(contents)
        extracted_texts = re.findall(r'【(.*?)】', formatted_text)
        category = extracted_texts[0] if extracted_texts else "Unknown"
        print(category)

        output_lines.append(f"{query} ### {category}")
        print(output_lines)

        with open("training_data.txt", "a", encoding="utf-8") as file:
            file.write("\n".join(output_lines) + "\n")

    else:
        print("Error:", response.status_code, response.text)

