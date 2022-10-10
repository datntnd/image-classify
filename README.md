If run in new project, we need change in some files: 
- Change url minio in core/settings/development.py
- Change mlflow url in compare.py and train.py

In here, we have too many config:

- File config.json have: 

    - service_name: name bentoml serving 
    - endpoint: name function in service.py

- In system has env: 
    - project_id: 
    - serving_host
    - serving_port
