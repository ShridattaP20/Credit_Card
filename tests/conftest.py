import pytest
import yaml
import os
import json


params_path = "params.yaml"

@pytest.fixture
def config(config_path=params_path):
    with open(config_path) as yaml_file:
        config = yaml.safe_load(yaml_file)
    return config
