import requests
# This is delibrately added since I have shifted from scikit learn 1.2.1 to 1.2.2 since deployments started
import warnings
warnings.filterwarnings("ignore")


ride = {
    "PULocationID": 10, 
    "DOLocationID": 50, 
    "trip_distance": 0
}

url = 'http://127.0.0.1:9696/predict'

response = requests.post(url, json=ride)
print(response.json())