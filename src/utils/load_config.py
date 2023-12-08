import json


def load_training_config():
  with open('config/training_config.json', 'r') as f: 
    config = json.load(f)
    return config