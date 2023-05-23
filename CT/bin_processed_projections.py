import numpy as np
import os
import sys
import time
import tqdm
import matplotlib.pyplot as plt
import argparse
import torch
import h5py
import pickle as pkl
import torch.nn.functional as F


parser = argparse.ArgumentParser(
    description="Binning of processed projections for CT reconstruction and GAN enhancement"
)

parser.add_argument("-root", type=str, required=True, help="root path")
parser.add_argument(
    "-oroot", type=str, default="", required=False, help="output root path"
)
parser.add_argument("-expname", type=str, required=True, help="Experiment name")
parser.add_argument("-copyname", type=str, required=True, help="Copy name")
parser.add_argument("-bin", type=int, default=2, required=True, help="Binning factor")


def main():
    args, unparsed = parser.parse_known_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    root = args.root
    oroot = args.oroot
    expname = args.expname
    copyname = args.copyname
    binning = args.bin

    # obj = EquinorDynamicCT(root, expname, oroot, expname)

    with h5py.File(os.path.join(root, f"{copyname}.h5"), "w") as o:
        with h5py.File(os.path.join(root, f"{expname}.h5"), "r") as f:
            # f.copy(f, o)
            # Not all meta information will be up to date, but not used.
            # Some hard coding due to using two environments, and use pytorch for binning.
            idxs = list(f["projections"].keys())

            f.copy(f["meta"], o, "meta")
            f.copy(f["angles"], o, "angles")

            o.create_group("projections")

            for i, idx in enumerate(tqdm.tqdm(idxs)):
                data = np.squeeze(f["projections"][idx])
                data = torch.from_numpy(data).to(device)
                data = F.avg_pool2d(data, binning).cpu().numpy()

                # del o["projections"][f"{idx}"]
                o["projections"].create_dataset(f"{idx}", data=data)

    geom_root = os.path.join(root, f"{expname}.pkl")
    with open(geom_root, "rb") as g:
        geo = pkl.load(g)

    geo.nDetector = geo.nDetector // binning
    geo.dDetector = geo.dDetector * binning

    geo.nVoxel = geo.nVoxel // binning
    geo.dVoxel = geo.dVoxel * binning

    with open(os.path.join(root, f"{copyname}.pkl"), "wb+") as g:
        pkl.dump(geo, g)

    return


if __name__ == "__main__":
    main()
