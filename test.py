import requests

s = [{"role": "user", "content": "Write a short essay on China"}]
parameters = {
    "max_new_tokens": 256,
    "temperature": 0.6,
    "top_p": 0.95,  
    "apply_chat_template": True
}

response = requests.post("http://127.0.0.1:8002/api/generate", json={"inputs": s, "parameters": parameters})
print(response.json()['generated_text'])