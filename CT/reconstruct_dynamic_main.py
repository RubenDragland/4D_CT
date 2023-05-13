from data_making import *
import numpy as np
import os
import sys
import time
import tqdm
import matplotlib.pyplot as plt


def main():
    root = r"/media/disks/disk2/CT-data/rubensd/Processed_projections/"
    expnames = [
        # "hourglassV3_13_55",
        "limestoneV3_17_55_6favg",
        "limestoneV3_17_55_12favg",
        "limestoneV3_17_55_24favg",
        # "limestoneV3_1440_1_std",
    ]
    oroot = r"/home/rubensd/Documents/DeepLearning/ReconstructionData"

    fibonaccis = [1, 1, 1, 55, 1]
    name = "Fibonacci1"

    for i in tqdm.trange(len(expnames)):
        expname = expnames[i]
        print("Reconstructing", expname)
        model = EquinorDynamicCT(root, expname, oroot, expname)

        if fibonaccis[i] == 1:
            rec = model.reconstruct_idx(idx=0, CoR=0)

            model.save_custom(
                rec, name=name, idx=0, fibonacci=fibonaccis[i], method="fdk"
            )
        else:
            model.reconstruct_custom(idx=0, fibonacci=fibonaccis[i], name=name, CoR=0)


if __name__ == "__main__":
    main()
