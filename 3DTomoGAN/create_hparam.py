import numpy as np
import pickle as pkl
import torch
import json
import os

hparams = {
    "name": "sirt_training_continued",
    "lmse": 50,
    "ladv": 100,
    "lperc": 1,
    "psz": 128,
    "mbsz": 1,
    "itg": 200,
    "itd": 1,
    "lrateg": 1e-5,
    "lrated": 2e-5,
    "train_split": 0.88,
    "transforms": "basic",
    "transfer_model": "SIRT_basic2_it00100_gen",
}

if __name__ == "__main__":
    my_path = os.path.abspath(os.path.dirname(__file__))

    with open(os.path.join(my_path, "hparams", f"{hparams['name']}.json"), "w+") as d:
        json.dump(hparams, d)

    with open(os.path.join(my_path, "hparams", f"{hparams['name']}.json"), "r") as d:
        dic = json.load(d)

    print(dic)
