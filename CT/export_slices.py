import data_making as dm
import numpy as np
import os
import sys
import time
import tqdm
from multiprocessing import Pool

root = r"/media/disks/disk2/CT-data/rubensd/Processed_projections/"

names = [
    # "hourglassV3_13_55",
    # "limestoneV3_17_55_6favg",
    "limestoneV3_17_55_12favg",
    # "limestoneV3_17_55_24favg",
    # "limestoneV3_1440_1_std",
]

oroot = r"/home/rubensd/Documents/DeepLearning/ReconstructionData"

# x1, x2 = 0, 256  # 672, 928
# y1, y2 = 0, 256  # 384, 640
# z1, z2 = 0, 256  # 384, 640

x1, x2 = 572, 828  # 672, 928
y1, y2 = 300, 556  # 384, 640
z1, z2 = 600, 856  # 384, 640

slice = [[x1, x2], [y1, y2], [z1, z2]]

identifiers = [
    # "gt",
    # "Fibonacci1",
    # "Fibonacci2",
    # "Fibonacci3",
    # "aligned",
    # "Rec_0_1_fdk",
    # "Rec_0_3_fdk",
    # "Rec_0_5_fdk",
    # "Rec_0_8_fdk",
    # "Rec_0_13_fdk",
    # "Rec_0_1_fdk_enhanced_700428728",
    # "Rec_0_3_fdk_enhanced_700428728",
    # "Rec_0_5_fdk_enhanced_700428728",
    # "Rec_0_8_fdk_enhanced_700428728",
    # "Rec_0_13_fdk_enhanced_700428728",
    "Rec_0_1_sirt",
    # "Rec_0_3_sirt"
    # # _enhanced_800512512",  # _enhanced_800512512",
    # "Rec_0_5_fdk_enhanced_800512512",
    # "Rec_0_8_fdk_enhanced_800512512",
]

# Ignore parallel. Should take 0 seconds anyway.
for i in range(len(names)):
    for j in range(len(identifiers)):
        obj = dm.EquinorDynamicCT(root, names[i], oroot, names[i])

        img = obj.load_4plot_slice(
            name=identifiers[j],
            idxs=slice,
        )

        save_name = f"{names[i]}_{identifiers[j]}_{x1}_{x2}_{y1}_{y2}_{z1}_{z2}.npy"
        np.save(save_name, img)
