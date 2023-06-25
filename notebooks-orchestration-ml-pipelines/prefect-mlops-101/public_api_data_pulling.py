import httpx
from prefect import flow

@flow
def get_data_from_api():
    api_url = "https://api.apis.guru/v2/list.json"
    response = httpx.get(api_url)
    
    email = response.json()["lforge.com"]["added"]
    print(email.text)

if __name__ == "__main__":
    get_data_from_api()