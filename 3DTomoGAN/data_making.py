import os
import sys
import time
import tqdm
import numpy as np
import torch
import imageio
import pandas as pd
import h5py

# RSD: Reading raw file
def read_raw_data(parent, filename, dat_parent, dat_filename):

    # dat_file = pd.read_table(os.path.join(dat_parent, dat_filename))

    # w, h, d = (
    #     dat_file["width"][0],
    #     dat_file["height"][0],
    #     dat_file["depth"][0],
    # )  # RSD: ISH
    w, h, d = 512, 512, 512  # RSD: ISH

    full = os.path.join(parent, filename)
    with open(full, "rb") as f:
        data = np.fromfile(f, dtype=np.float32)

    # data = data.reshape((d, h, w))

    return data


def get_dataset_keys(f):
    keys = []
    f.visit(lambda key: keys.append(key) if isinstance(f[key], h5py.Dataset) else None)
    return keys


def produce_3D_sinogram():  # RSD: Alternatively, produce 3D from a single reconstruction?
    """
    Takes TomoBank data, especially phantoms, and expand them in another dimension. save as h5 file.
    """
    return


def produce_3D_reconstruction():
    """
    Takes TomoBank data, especially phantoms, and expand them in another dimension. save as h5 file.
    """
    return
