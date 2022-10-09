export GRAFANA_SERVER=10.61.185.119:3000
export GRAFANA_API_KEY=eyJrIjoiQ3BvSnVaamYydEF0UkpFV3dLQkgxRmdUZ24xWUtMREMiLCJuIjoia2luZ2tvbmciLCJpZCI6MX0=


bash upload.sh  data.json

#kong_address
python upload_kong.py
