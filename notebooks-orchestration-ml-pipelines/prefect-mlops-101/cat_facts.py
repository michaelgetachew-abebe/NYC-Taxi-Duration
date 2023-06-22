import httpx
from prefect import task, flow

@task(retries=4, retry_delay_seconds=0.1, log_prints=True)
def fetch_cat_facts():
    dummy_cat_url = "https://f3-vyx5c2hfpq-ue.a.run.app/"
    cat_fact = httpx.get(dummy_cat_url)
    #An endpoint that is designed to fail sporadically
    if cat_fact.status_code >= 400:
        raise Exception
    print(cat_fact.text)

@flow
def fetch():
    fetch_cat_facts()

if __name__ == "__main__":
    fetch()
