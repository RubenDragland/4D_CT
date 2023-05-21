import numpy as np
import pickle as pkl
import torch
import json
import os

hparams = {
    "name": "sim_data_continued",
    "lmse": 5,
    "ladv": 10,
    "lperc": 0,
    "psz": 128,
    "mbsz": 1,
    "itg": 2,
    "itd": 1,
    "lrateg": 1e-4,
    "lrated": 2e-4,
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
