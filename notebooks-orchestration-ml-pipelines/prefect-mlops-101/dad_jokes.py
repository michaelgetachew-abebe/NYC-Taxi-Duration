import config
import httpx
from prefect import flow, task

@task(retries=4, retry_delay_seconds=0.1, log_prints=True)
def pull_dad_jokes():
    api_url = "https://api.api-ninjas.com/v1/dadjokes?limit="
    response = httpx.get("https://api.api-ninjas.com/v1/dadjokes?limit=2", headers={'X-Api-Key': config.api_key}).json()[0]

    print(response["joke"])

@flow
def make_me_laugh():
    for i in range(3):
        pull_dad_jokes()


if __name__ == "__main__":
    make_me_laugh()