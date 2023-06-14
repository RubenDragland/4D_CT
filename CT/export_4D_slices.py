import data_making as dm
import numpy as np
import os
import sys
import time
import tqdm
from multiprocessing import Pool

root = r"/media/disks/disk2/CT-data/rubensd/Processed_projections/"

names = ["hourglassV3_13_55_2bin"]

oroot = r"/home/rubensd/Documents/DeepLearning/ReconstructionData"


x1, x2 = 0, 448
y1, y2 = 0, 264
z1, z2 = 0, 264


slice = [[x1, x2], [y1, y2], [z1, z2]]

identifiers = ["4D_4_fdk"]

indices = [x for x in range(0, 49, 4)]

# Ignore parallel. Should take 0 seconds anyway.
for i in range(len(names)):
    for j in range(len(identifiers)):
        for idx in indices:
            obj = dm.EquinorDynamicCT(root, names[i], oroot, names[i])

            img = obj.load_4plot_slice(
                name=identifiers[j],
                idxs=slice,
                idx=f"{str(idx).zfill(5)}_enhanced_442256256",
            )

            save_name = f"{names[i]}_{identifiers[j]}_{str(idx).zfill(5)}_{x1}_{x2}_{y1}_{y2}_{z1}_{z2}.npy"
            np.save(save_name, img)
