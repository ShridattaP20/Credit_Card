import yaml
import os
import json
import joblib
import numpy as np


params_path = "params.yaml"


def read_params(config_path):
    with open(config_path) as yaml_file:
        config = yaml.safe_load(yaml_file)
    return config


def predict(data):
    config = read_params(params_path)
    model_dir_path = config["webapp_model_dir"]
    model = joblib.load(model_dir_path)
    prediction = model.predict(data).tolist()[0]
    print(prediction)
    return prediction


def form_response(dict_request):
    data = dict_request.values()
    data = [list(map(float, data))]
    response = predict(data)
    return response


def api_response(dict_request):
    try:
        data = np.array([list(dict_request.values())])
        response = predict(data)
        response = {"response": response}
        return response

    # except NotInRange as e:
    #     response = {"the_exected_range": get_schema(), "response": str(e) }
    #     return response

    # except NotInCols as e:
    #     response = {"the_exected_cols": get_schema().keys(), "response": str(e) }
    #     return response

    except Exception as e:
        response = {"response": str(e)}
        return response
