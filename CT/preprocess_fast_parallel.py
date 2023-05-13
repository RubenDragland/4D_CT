from preprocessing import *
from multiprocessing import Pool

import os, sys
import time
import tqdm


def func(root, expname, oroot, num_proj, nrevs, correction, geometry, roi):
    preprocess = DynamicProjectionsEQNR(
        root,
        expname,
        oroot,
        num_proj,
        nrevs=nrevs,
        correction_parent=correction,
        geometry=geometry,
        roi=roi,
    )

    preprocess()


def main():
    main_root = r"/media/nfs/qnap/home/rubensd/RSD20230509"
    names = [
        "RSD20230509_hourglassV3_13proj",
        "RSD20230509_samplingV3_sandstone_6favg",
        "RSD20230509_sandstoneV3_12favg",
        "RSD20230509_sandstoneV3_24favg",
    ]
    roots = [os.path.join(main_root, name) for name in names]
    roots.append(r"/media/nfs/qnap/home/rubensd/RSD20230428_standard_sandstone")
    main_oroot = r"/media/disks/disk2/CT-data/rubensd/Processed_projections/"
    expnames = [
        "hourglassV3_13_55",
        "limestoneV3_17_55_6favg",
        "limestoneV3_17_55_12favg",
        "limestoneV3_17_55_24favg",
        "limestoneV3_1440_1_std",
    ]
    correction = r"/media/nfs/qnap/home/rubensd/20230424RSD/RSD20230424_samplingV3_hourglass/Corrections"
    num_proj = [13, 17, 17, 17, 1440]
    nrevs = [55, 55, 55, 55, 1]
    geoms = ["golden_motion.nsiprg"] * 4
    geoms.append(None)
    roi = [1536, 1024]

    args = [
        [
            roots[i],
            expnames[i],
            main_oroot,
            num_proj[i],
            nrevs[i],
            correction,
            geoms[i],
            roi,
        ]
        for i in range(len(roots))
    ]

    with Pool() as pool:
        pool.starmap(func, args)


if __name__ == "__main__":
    main()
