import requests

url = "http://localhost:31314/ml-image-processing"
payload = {"image_name": "000b7b74-0a22-4d0c-b717-e240fdc5d555.png"}

response = requests.post(url, json=payload)

print("Status code:", response.status_code)
print("Response body:", response.text)
