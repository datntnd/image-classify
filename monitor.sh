#export GRAFANA_SERVER=10.255.187.50:8000
#export GRAFANA_API_KEY=eyJrIjoiRllHWjIyc3ZDdnpKSXcyemRqZWNjdFhHT01RZTlaUG8iLCJuIjoic2VydmVyNDgiLCJpZCI6MX0=

bash upload.sh  data.json

#kong_address
kong_address=${kong_address:-10.255.187.46:8001}
serving_host=${serving_host:-10.61.185.119}
serving_port=${serving_port:-6000}

# export user_id=linhnt153 
# export serving_host=10.255.187.46
# export serving_port=6000 
# export kong_address=10.255.187.48:8001

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY


/home/anaconda3/bin/python3 upload_kong.py

