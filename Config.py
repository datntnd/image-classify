import json

config_file = "config.json"


def get_from_dict(field_name, dict_field, json_field):
    if (field_name in dict_field.keys()) is False:
        return json_field.get(field_name).get("default")
    elif dict_field[field_name] is None:
        return json_field.get(field_name).get("default")
    else:
        return dict_field[field_name]


class ConfigClassificationTrain:
    def __init__(self, config_dict):
        config = json.load(open(config_file))
        config = config.get("detail")
        config = config.get("train_config")
        self.model_name = get_from_dict("model-name", config_dict, config)
        self.pretrained_weights = get_from_dict("pretrained-weights", config_dict, config)
        self.train_data_path = get_from_dict("train-data-path", config_dict, config)
        self.val_data_path = get_from_dict("val-data-path", config_dict, config)
        self.epochs = get_from_dict("epochs", config_dict, config)
        self.batch_size = get_from_dict("batch-size", config_dict, config)
        self.lr = get_from_dict("lr", config_dict, config)
        self.num_classes = get_from_dict("num-classes", config_dict, config)
        self.freeze_layers = get_from_dict("freeze-layers", config_dict, config)
        self.service_name = get_from_dict("serve-bentoml-name", config_dict, config)
        self.endpoint = get_from_dict("endpoint", config_dict, config)


class ConfigClassificationPredict:
    def __init__(self, config_dict):
        config = json.load(open(config_file))
        config = config.get("detail")
        config = config.get("predict_config")
        self.model_name = get_from_dict("model-name", config_dict, config)
        self.pretrained_weights = get_from_dict("pretrained-weights", config_dict, config)
        self.data_path = get_from_dict("data-path", config_dict, config)
        self.num_classes = get_from_dict("num-classes", config_dict, config)
        self.batch_size = get_from_dict("batch-size", config_dict, config)

