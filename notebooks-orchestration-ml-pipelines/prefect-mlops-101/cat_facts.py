import httpx
from prefect import task, flow

@task(retries=4, retry_delay_seconds=0.1, log_prints=True)
def fetch_cat_facts():
    cat_fact = httpx.get("https://f3-vyx5c2hfpq-ue.a.run.app/")