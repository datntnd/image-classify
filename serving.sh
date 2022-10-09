serving_host=${serving_host:-10.61.185.119}
serving_port=${serving_port:-5000}

# Gen dashboard 
python3 gen-dashboard/gen_dashboard.py --instance $serving_host:$serving_port

# bash upload.sh  data.json

kill -9 $(lsof -t -i:$serving_port)


firewall-cmd --zone=public --add-port=$serving_port/tcp --permanent
firewall-cmd --reload

# bentoml serve service.py:svc --port $port
echo $serving_port
nohup /hdd/anaconda3/bin/bentoml serve service.py:svc --port $serving_port &

