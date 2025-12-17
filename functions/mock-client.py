import requests
import redis
import time
import json

r = redis.Redis(
    host="127.0.0.1",   # NOT "http://127.0.0.1"
    port=32204,
    decode_responses=True,
)


# url = "http://localhost:31314/ml-image-processing"
# payload = {"image_name": "000b7b74-0a22-4d0c-b717-e240fdc5d555.png"}
# url = "http://localhost:31314/ml-object-detection"
# payload = {"image_name": "000b7b74-0a22-4d0c-b717-e240fdc5d555_processed.png"}

# response = requests.post(url, json=payload)

def send_request(url, payload):
    response = requests.post(url, json=payload)
    res_payload = response.json()
    # print(f"payload: {payload}")
    print("Status code:", response.status_code)
    print("Response body:", response.json())
    req_id = res_payload.get("req_id", "N/A")
    return req_id


def poll_result(
    req_id: str,
    timeout_s: float = 5,
):
    deadline = time.monotonic() + timeout_s

    while True:
        val = r.get(req_id)
        if val is not None:
            # r.delete(result_key)
            return json.loads(val)

        if time.monotonic() >= deadline:
            raise TimeoutError(f"Timed out after {timeout_s:.1f}s waiting for {req_id}")

        time.sleep(0.5)


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
  
  # send_request(workflow["image_processing"]["url"], workflow["image_processing"]["payload"])
  # send_request(workflow["object_detection"]["url"], workflow["object_detection"]["payload"])
  req_id = send_request(workflow["image_processing"]["url"], workflow["image_processing"]["payload"])
  data = poll_result(req_id)
  print(data)

