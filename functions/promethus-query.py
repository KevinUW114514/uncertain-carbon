import requests

promethus_url = "http://localhost:30990/api/v1/query"

def send_request(url, query):
    response = requests.get(url, params={"query": query})
    print(response.json())
    print("Status code:", response.status_code)
    
def avg_p99_duration_20s(name):
    return f'http_requests_duration_seconds{{path={name}, quantile="0.99"}} and on (path) (increase(http_requests_duration_seconds_count{{path={name}}}[20s]) > 0)'

if __name__ == "__main__":
  

  send_request(promethus_url, avg_p99_duration_20s("ml-image-processing"))
  send_request(promethus_url, avg_p99_duration_20s("ml-object-detection"))