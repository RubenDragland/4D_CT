import torch
import h5py
import sys, os, argparse, shutil
import numpy as np
import tqdm
from models import Generator3DTomoGAN
import matplotlib.pyplot as plt
import torchio as tio
import utils

import inference

model_path = r"/home/rubensd/Documents/DeepLearning/Results/Models/"
model_name = "FINALFDK_it00092_gen"
data_folder = "/home/rubensd/Documents/DeepLearning/ReconstructionData/"
data_name = "hourglassV3_13_55_2bin"
key_inputs = [f"4D_4_fdk/{str(z).zfill(5)}" for z in range(0, 49, 4)]
key_target = "gt"
focus = [442, 256, 256]
dims = [448, 264, 264]


def main():
    for i, k in enumerate(tqdm.tqdm(key_inputs)):
        inference.enhance(
            model_path=model_path,
            model_name=model_name,
            data_folder=data_folder,
            data_name=data_name,
            key_input=k,
            key_target=key_target,
            focus=focus,
            dims=dims,
        )


if __name__ == "__main__":
    main()
