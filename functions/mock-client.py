import requests

# url = "http://localhost:31314/ml-image-processing"
# payload = {"image_name": "000b7b74-0a22-4d0c-b717-e240fdc5d555.png"}
# url = "http://localhost:31314/ml-object-detection"
# payload = {"image_name": "000b7b74-0a22-4d0c-b717-e240fdc5d555_processed.png"}

# response = requests.post(url, json=payload)

def send_request(url, payload):
    response = requests.post(url, json=payload)
    print("Status code:", response.status_code)
    print("Response body:", response.json())
    

if __name__ == "__main__":
  workflow = {
    "image_processing": {
      "url": "http://localhost:31314/ml-image-processing",
      "payload": {"image_name": "000b7b74-0a22-4d0c-b717-e240fdc5d555.png"}
    },
    "object_detection": {
      "url": "http://localhost:31314/ml-object-detection",
      "payload": {"image_name": "000b7b74-0a22-4d0c-b717-e240fdc5d555_processed.png"}
    }
  }
  
  send_request(workflow["image_processing"]["url"], workflow["image_processing"]["payload"])
  send_request(workflow["object_detection"]["url"], workflow["object_detection"]["payload"])
