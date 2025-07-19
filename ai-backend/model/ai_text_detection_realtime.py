import requests

API_URL = "https://api-inference.huggingface.co/models/roberta-base-openai-detector"
# Optionally, insert your Hugging Face token for higher rate limits
# headers = {"Authorization": "Bearer YOUR_HF_TOKEN_HERE"}  # <-- Insert your token here if needed
headers = {}  # No token, public rate limits

def query(text):
    response = requests.post(API_URL, json={"inputs": text}, headers=headers)
    try:
        return response.json()
    except Exception as e:
        return {"error": str(e), "raw_response": response.text}

print("Real-time AI Text Detection (Hugging Face API)")
print("Type/paste text and press Enter. Type 'exit' to quit.")

while True:
    user_input = input("\nEnter text: ")
    if user_input.strip().lower() == 'exit':
        break
    print("Checking... (may take a few seconds)")
    result = query(user_input)
    print("Result:", result)

# Instructions:
# 1. (Optional) Get a free Hugging Face token at https://huggingface.co/settings/tokens
# 2. Insert your token in the headers if you want higher rate limits.
# 3. Run: python ai_text_detection_realtime.py
# 4. Enter text to check if it's AI-generated in real time. 