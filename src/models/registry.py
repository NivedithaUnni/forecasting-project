import json
import os

REGISTRY_PATH = "models/model_registry.json"


class ModelRegistry:

    def __init__(self):
        if not os.path.exists(REGISTRY_PATH):
            with open(REGISTRY_PATH, "w") as f:
                json.dump({}, f)

    def save_best_model(self, state, best_model_name, mae):
        with open(REGISTRY_PATH, "r") as f:
            data = json.load(f)

        data[state] = {
            "best_model": best_model_name,
            "mae": mae
        }

        with open(REGISTRY_PATH, "w") as f:
            json.dump(data, f, indent=4)

    def get_best_model(self, state):
        with open(REGISTRY_PATH, "r") as f:
            data = json.load(f)

        return data.get(state, None)