# example terminal chatbot program

# input: text
# output: text


import requests

url = "http://localhost:8000/generate"
request = {"text": "What is the capital of France?"}

response = requests.post(
    url=url,
    json=request,
)

print(response.json())

