import requests
import json
from core.config import get_app_settings
import os
from Config import ConfigClassificationTrain

settings = get_app_settings()
config = ConfigClassificationTrain({})

user_id = settings.user_id
project_id = settings.project_id
serving_host = os.environ.get("serving_host")
serving_port = os.environ.get("serving_port")
kong_address = settings.kong_address

print(f"kong address: {kong_address}")


# Create services

url_service = f"http://{kong_address}/services/"

headers = {
  'Content-Type': 'application/json'
}

name_service = config.service_name + "_" + user_id

payload_services=json.dumps({
  "name": name_service,
  "host": serving_host,
  "port": int(serving_port),
  "path": config.endpoint,
  "protocol": "http"
})

print(payload_services)

response = requests.request("POST", url_service, headers=headers, data=payload_services)

print(response.text)

#Create routes

url = f"http://{kong_address}/services/{name_service}/routes"
name_route = f"/{project_id}/iva-model"

payload_routes = json.dumps({
  "name": name_service,
  "paths": [ name_route ]
})


response = requests.request("POST", url, headers=headers, data=payload_routes)

print(response.text)
