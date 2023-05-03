import numpy as np
import pickle as pkl
import torch
import json
import os 

hparams = {
    "name": "std",
    "lmse": 10,
    "ladv": 1,
    "psz": 128,
    "mbsz": 1,
    "itg": 1,
    "itd": 1,
    "lrateg": 1e-4,
    "lrated": 1e-4,
    "train_split": 0.9,
    "transforms": "basic"
}

if __name__ == "__main__":

    my_path = os.path.abspath(os.path.dirname(__file__))

    with open(os.path.join(my_path, "hparams", f"{hparams['name']}.json"), "w+") as d:

        json.dump(hparams, d)

    with open(os.path.join(my_path, "hparams", f"{hparams['name']}.json"), "r") as d:

        dic = json.load(d)

    print(dic)