import pickle
import json


def load_langvec(path):
    with open(path, "rb") as file:
        langvec = pickle.load(file)
    return langvec


def save_pkl(data, path):
    with open(path, "wb") as file:
        pickle.dump(data, file)


def save_json(data, path):
    json.dumps(data, sort_keys=True)
