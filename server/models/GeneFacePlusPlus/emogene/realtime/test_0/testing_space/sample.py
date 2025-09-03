import requests

request = requests.get("http://localhost:8000")

print(request.json())

request = requests.get("http://localhost:8000/random")

print(request.json())

request = requests.get("http://localhost:8000/random/50")

print(request.json())