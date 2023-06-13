import numpy as np
import pickle as pkl
import torch
import json
import os

hparams = {
    "name": "WGAN",
    "lmse": 1,
    "ladv": 10,
    "lperc": 0,
    "psz": 80,
    "mbsz": 4,
    "itg": 10,
    "itd": 10,
    "lrateg": 1e-5,
    "lrated": 2e-5,
    "train_split": 0.88,
    "transforms": "basic",
    "transfer_model": "simV1_it00500_gen",
}

if __name__ == "__main__":
    my_path = os.path.abspath(os.path.dirname(__file__))

    with open(os.path.join(my_path, "hparams", f"{hparams['name']}.json"), "w+") as d:
        json.dump(hparams, d)

    with open(os.path.join(my_path, "hparams", f"{hparams['name']}.json"), "r") as d:
        dic = json.load(d)

    print(dic)
