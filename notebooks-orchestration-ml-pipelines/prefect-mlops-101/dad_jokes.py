import config
import httpx
from prefect import flow

@task
def pull_dad_jokes():
    api_url = "https://api.api-ninjas.com/v1/dadjokes?limit="
    response = httpx.get("https://api.api-ninjas.com/v1/dadjokes?limit=", headers={'X-Api-Key': config.api_key}).json()["joke"]

    return response

@flow
def make_me_laugh():
    for i in range(3):
        pull_dad_jokes()


if __name__ == "__main__":
    make_me_laugh()