from string import Template
import json
import sys, os
sys.path.append(os.path.abspath('.'))
import configargparse
from Config import ConfigClassificationTrain
from core.config import get_app_settings
import os

settings = get_app_settings()
user_id = settings.user_id
project_id = settings.project_id

print("gen dashboard")
print(f"user_id: {user_id}")
print(f"project_id: {project_id}")

def gen_dashboard_json(data):
    with open('gen-dashboard/template.json', 'r') as json_file:
        content = ''.join(json_file.readlines())
        template = Template(content)
        configuration = json.loads(template.substitute(data))
        # print(configuration)

    with open('data.json', 'w') as f:
        json.dump(configuration, f)


if __name__ == '__main__':
    parser = configargparse.ArgumentParser(
        description="Dashboard config parser",
        config_file_parser_class=configargparse.YAMLConfigFileParser,
        formatter_class=configargparse.ArgumentDefaultsHelpFormatter,
    )
    # general configuration
    # name is actually in bentoml.Service(name="pytorch_flower_demo", runners=[mnist_runner])
    # parser.add("--name", type=str, default="pytorch_flower_demo", help="config name dashboard")
    parser.add("--instance", type=str, default="10.61.185.119:5000", help="config server dashboard")
    # parser.add("--endpoint", type=str, default="/predict_image", help="config endpoint dashboard")
    config = ConfigClassificationTrain({})

    args = parser.parse_args()
    print(f"args: {args}")

    title = f"image_classify_{user_id}_project_id_{project_id}_{args.instance}" 

    data = dict(
        name=f"image_classify_{user_id}_project_id_{project_id}",
        instance=args.instance,
        endpoint=config.endpoint,
        title=title,
        __rate_interval='$__rate_interval'
    )
    gen_dashboard_json(data)
