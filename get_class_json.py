import glob
import json
import os
from pathlib import Path


def get_classes_json(folder="data/", json_file_name="classes.json"):
    classes = {}
    # if not os.path.exists(f"data/{json_file_name}"):
    i = 0
    for phase in ['training', 'testing', 'validation' ]:
        path_label = folder + phase
        for file in glob.glob(path_label + "/labels/*.txt"):
            label_file = open(file, "r")
            label = label_file.readline().strip('\n')
            if not (label in classes.keys()):
                classes.update({
                    label: str(i)
                })
                i += 1

    with open(f'data/{json_file_name}', 'w', encoding='utf-8') as f:
        json.dump(classes, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    get_classes_json(folder="data/")
