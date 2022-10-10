serving_host=${serving_host:-10.255.187.48}
serving_port=${serving_port:-1234}

# Gen dashboard 
/home/anaconda3/bin/python3 gen-dashboard/gen_dashboard.py --instance $serving_host:$serving_port

# bash upload.sh  data.json

kill -9 $(lsof -t -i:$serving_port)


firewall-cmd --zone=public --add-port=$serving_port/tcp --permanent
firewall-cmd --reload

# bentoml serve service.py:svc --port $port
echo $serving_port
nohup /home/anaconda3/bin/bentoml serve service.py:svc --port $serving_port &

