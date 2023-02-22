import os
import sys
import time
import tqdm
import numpy as np
import torch

# import imageio
import pandas as pd
import h5py
import tigre
import tigre.algorithms as algs
import tigre.utilities.gpu as gpu

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


def process_raw_file_reconstruction(parent, filenames, dat_parent, dat_filenames):
    # RSD: Process raw files and save as h5 file.
    # RSD: Need to figure out how to do reconstruction.

    for i, (filename, dat) in enumerate(zip(filenames, dat_filenames)):
        data = read_raw_data(parent, filename, dat_parent[i], dat_filenames[i])
        # RSD: Back to sinogram?
        # Save sinogram, and target as h5 file.
        # RSD: Save as h5 file.
    return


# RSD: File for reading tiff files and doing reconstructions from radiographs
def process_radiographs():

    # RSD: Read tiff files
    # Create Sinograms
    # Storing sinograms
    # Slicing data, keeping gt
    # Reconstruction
    # Save as h5 file
    return


def get_dataset_keys(f):
    keys = []
    f.visit(lambda key: keys.append(key) if isinstance(f[key], h5py.Dataset) else None)
    return keys


# def merge_h5(parent, names, output, oname):
#     """
#     Takes prepared hdf5 files and merges them into one.
#     """
#     o = h5py.File(os.path.join(output, f"{oname}.h5"), "w")
#     for i, name in enumerate(names):

#     return

# RSD: Should have some uint8 conversion as well. Dunno.
def fix_dtype():
    return


def sinogram_2_datafile(
    parent, names, output, oname, depth=256, keys=["exchange", "data"]
):  # RSD: Alternatively, produce 3D from a single reconstruction?
    """
    Takes TomoBank data, especially phantoms, and expand them in another dimension. save as h5 file.
    Necessary with flatfield correction etc?
    """

    o = h5py.File(os.path.join(output, f"{oname}.h5"), "w")
    o.create_group("sinogram3D")

    for i, name in enumerate(names):
        f = h5py.File(os.path.join(parent, f"{name}.h5"), "r")
        sino = np.squeeze(f[keys[0]][keys[1]])
        sino = np.broadcast_to(
            sino, (sino.shape[0], sino.shape[1], depth)
        )  # Leave depth to be the last dimension
        f.close()

        # RSD: Save as h5 file. Await the reconstruction step. Have to figure out how.

        o["sinogram3D"].create_dataset(f"{i}", data=sino)  # Or use name?
    o.close()
    return


def target_2_datafile(parent, names, output, depth=256, keys=["exchange", "gt"]):
    """
    Takes TomoBank data, especially phantoms, and expand the ground truth in another dimension. Writes to h5 file.
    """
    o = h5py.File(os.path.join(output, f"{name}3D.h5"), "w")
    o.create_group("target3D")
    for i, name in enumerate(names):

        f = h5py.File(os.path.join(parent, f"{name}.h5"), "r")
        target = np.squeeze(f[keys[0]][keys[1]])
        target = np.broadcast_to(
            target, (target.shape[0], target.shape[1], depth)
        )  # Leave depth to be the last dimension
        f.close()

        # RSD: Save as h5 file.

        o["target3D"].create_dataset(
            f"i", data=target
        )  # Or use name? Check with datafile class.
    o.close()
    return


def phantom_sinogram_reconstruction(
    parent,
    output,
    num_projections,
    dim_voxels=[512, 512, 512],
):
    """
    Perform reconstruction and save to h5 file.
    Assumes h5 file with sinogram3D and target3D.
    Reconstructs directly using the sinogram as projections.
    Should be copied and be performed with more and fewer projections. Add on later.
    Would simply be to repeat the process with different number of projections and somehow merge even though indexing becomes an issue.

    Not sure if this works.
    """
    listGpuNames = gpu.getGpuNames()
    gpuids = gpu.getGpuIds(listGpuNames[0])

    o = h5py.File(os.path.join(parent, f"{output}.h5"), "w")
    o.create_group("noisy3D")

    f = h5py.File(os.path.join(parent, f"{output}.h5"), "r")

    sino_geo = tigre.geometry(mode="cone", default=True, nVoxel=dim_voxels)
    angles = np.linspace(0, 2 * np.pi, num_projections, endpoint=False)

    for i, sinogram3D in enumerate(f["sinogram3D"]):

        rec = algs.fdk(sinogram3D, sino_geo, angles, gpuids=gpuids)

        o["noisy3D"].create_dataset(f"{i}", data=rec)

    f.close()
    o.close()

    return


def undersampled_reconstruction():
    """
    Figure out what to do with this thing. For now, one cannot do cropping on the fly while training. It would require to do the entire reconstruction on the fly.
    Instead, choose random number of projections at (random angles)?, and reconstruct and save to hdf5 file.
    Also, figure out the environments. Transfer CIL-environment to 4DCT since they are bound to work together. If not, some think will have to be conducted by each environment.
    """
    return


def phantom_reconstruction(parent, output, n_projections, dim_voxels=[512, 512, 512]):
    """
    Reconstruction of phantoms.
    Assumes h5 file with sinogram3D and target3D.
    Skips the use of sinogram. Or is projections the same as sinogram? Should be.
    """

    listGpuNames = gpu.getGpuNames()
    gpuids = gpu.getGpuIds(listGpuNames[0])

    o = h5py.File(os.path.join(parent, f"{output}.h5"), "w")
    o.create_group("noisy3D")

    f = h5py.File(os.path.join(parent, f"{output}.h5"), "r")

    sino_geo = tigre.geometry(mode="cone", default=True, nVoxel=dim_voxels)
    angles = np.linspace(0, 2 * np.pi, n_projections, endpoint=False)

    for i, target3D in enumerate(f["target3D"]):

        projs = tigre.Ax(target3D, sino_geo, angles, gpuids=gpuids)

        rec = algs.fdk(projs, sino_geo, angles, gpuids=gpuids)

        o["noisy3D"].create_dataset(f"{i}", data=rec)

    f.close()
    o.close()

    return


# RSD: SHould possibly the expansion be done when doing dataloading?
# RSD: Plan: Create dataset.
"""

Take phantoms, expand them in another dimension, save the target in h5 file. Also need undersampled data. 
Reconstruction on the fly would save disk space, but will be compuationally expensive
For phantoms, a simple inverse radon transform should be enough after slicing the sinogram? Or possibly necessary to use tomophantom??? Check this out.
Reconstruct with different number of projections? Or possibly a given number of projections first. Vary what angles. 

For experimental data, read the raw file, crop volume into smaller volumes, save target. Use TomoPhantom to generate sinogram, slice and reconstruct undersampled training data. 
Will have to check this first idea regardless. But a volume is a volume, så should be in the clear. 
Or should one create create radiographs, crop, slice and reconstruct? Does not make sense though...

Cropping of volumes may be done on the fly though. 


"""
