import httpx
from prefect import flow

@flow
def get_repo_info():
    repo_url = "https://api.github.com/repos/PrefectHQ/prefect"
    response = httpx.get(repo_url)
    response.raise_for_status()
    repo = response.json()
    print(f"PrefectHQ/prefect repository statistics 🤓:")
    print(f"Stars 🌠 : {repo['stargazers_count']}")
    print(f"Forks 🍴 : {repo['forks_count']}")


if __name__ == "__main__":
    get_repo_info()