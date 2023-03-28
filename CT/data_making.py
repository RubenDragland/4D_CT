import os
import sys
import time
import tqdm
import numpy as np
import pickle as pkl

# import imageio
import pandas as pd
import h5py
import tigre
import tigre.algorithms as algs
import tigre.utilities.gpu as gpu
import matplotlib.pyplot as plt


class ReconstructionsDataCT:

    NOISY_KEY = "noisy3D"
    TARGET_KEY = "target3D"
    SINO_KEY = "sinogram3D"

    TOMOBANK_SINO = "exchange/data"
    TOMOBANK_TARGET = "exchange/ground_truth"
    TOMOBANK_ANGLES = "exchange/theta"

    EQNR_PROJECTIONS = "projections"
    EQNR_ANGLES = "angles"
    EQNR_GEOMETRY = "geometry"

    def __init__(self, data_root, data_name):
        self.data_root = data_root
        self.data_name = data_name
        self.full_path = os.path.join(self.data_root, f"{self.data_name}.h5")

        try:
            o = h5py.File(self.full_path, "w-")
        except:
            o = h5py.File(self.full_path, "r+")

        # o = h5py.File(os.path.join(self.data_root, f"{self.data_name}.h5"), "a+")
        try:
            isinstance(o[ReconstructionsDataCT.NOISY_KEY], h5py.Group)
        except:
            o.create_group(ReconstructionsDataCT.NOISY_KEY)
        try:
            isinstance(o[ReconstructionsDataCT.TARGET_KEY], h5py.Group)
        except:
            o.create_group(ReconstructionsDataCT.TARGET_KEY)
        try:
            isinstance(o[ReconstructionsDataCT.SINO_KEY], h5py.Group)
        except:
            o.create_group(ReconstructionsDataCT.SINO_KEY)

        o.close()

    def add_item(self, obj):
        o = h5py.File(self.full_path, "r+")
        f = h5py.File(os.path.join(obj.root, f"{obj.name}.h5"), "r")
        data = obj.reconstruct_target(o, f, self.__len__())
        obj.reconstruct_noisy(o, f, data, self.__len__())
        f.close()
        o.close()
        return

    def __len__(self):
        return len(h5py.File(self.full_path)[self.NOISY_KEY].keys())

    def process_data(self, objects: list):

        o = h5py.File(self.full_path, "r+")

        for i, obj in enumerate(objects):

            index = self.__len__()
            f = h5py.File(os.path.join(obj.root, f"{obj.name}.h5"), "r")
            data = obj.reconstruct_target(o, f, index)
            obj.reconstruct_noisy(o, f, data, index)
            f.close()

        o.close()
        return


class PhantomGeometry:
    def __init__(self, default=True):

        if default:

            self.values = {
                "DSD": 1350,  # Distance Source Detector (mm)
                "DSO": 930,  # Distance Source Origin (mm)
                "nDetector": np.array([1024, 512]),  # number of pixels (px)
                "dDetector": np.array([0.2, 0.2]),  # size of each pixel (mm)
                "rotation": np.array([0, 0, 0]),  # Rotation of detector
            }

            # self.values = {
            #     "DSD": 1350,  # Distance Source Detector (mm)
            #     "DSO": 1300,  # Distance Source Origin (mm)
            #     "nDetector": np.array([1024, 512]),  # number of pixels (px)
            #     "dDetector": np.array([0.2, 0.2]),  # size of each pixel (mm)
            #     "rotation": np.array([0, 0, 0]),  # Rotation of detector
            # }

        else:
            self.phan_geo = tigre.geometry(
                mode="cone", default=True, nVoxel=[256, 256, 256]
            )
            return

        phan_geo = tigre.geometry(mode="cone", default=True)

        phan_geo.DSD = self.values["DSD"]  # Distance Source Detector (mm)
        phan_geo.DSO = self.values["DSO"]  # Distance Source Origin (mm)

        phan_geo.nDetector = self.values["nDetector"]  # number of pixels (px)
        phan_geo.dDetector = self.values["dDetector"]  # size of each pixel (mm)
        phan_geo.sDetector = (
            phan_geo.dDetector * phan_geo.nDetector
        )  # total size of the detector (mm)

        phan_geo.nVoxel = np.array(
            [512, 256, 256]
        )  # Fix this to fit with the reconstruction class.
        phan_geo.dVoxel = np.repeat(phan_geo.dDetector[0], 3)  # size of each voxel
        phan_geo.sVoxel = phan_geo.dVoxel * phan_geo.nVoxel  # total size of the image

        phan_geo.offOrigin = np.array([0, 0, 0])  # Offset of image from origin (mm)
        phan_geo.offDetector = np.array([0, 0])  # Offset of Detector

        phan_geo.accuracy = 0.5  # Accuracy of FWD proj    (vx/sample)

        phan_geo.COR = 0

        phan_geo.rotDetector = self.values["rotation"]  # Rotation of detector

        self.phan_geo = phan_geo

        return

    def __call__(self):
        return self.phan_geo


class TomoBankPhantomCT(ReconstructionsDataCT):
    def __init__(self, root, name, o_root, o_name, sino_recon=True):
        super().__init__(o_root, o_name)
        self.root = root
        self.name = name

    def reconstruct_target(self, o, f, idx, depth=512, sino=False):

        if sino:  # RSD: Ignored for now. Not much to gain.
            angles = np.squeeze(
                np.array(f[ReconstructionsDataCT.TOMOBANK_ANGLES])
            )  # np.squeeze(f[ReconstructionsDataCT.EQNR_ANGLES])
            data = np.squeeze(f[ReconstructionsDataCT.TOMOBANK_SINO])
            o[ReconstructionsDataCT.SINO_KEY].create_dataset(
                f"{str(idx).zfill(5)}", data=data
            )
            # RSD: Some lines of code for reconstruction are lacking if sino is to be used.
        else:
            data = np.squeeze(f[ReconstructionsDataCT.TOMOBANK_TARGET]).astype(
                np.float32
            )[np.newaxis, :, :]

            data = np.vstack([data] * depth)
            print(data.shape)
            o[ReconstructionsDataCT.TARGET_KEY].create_dataset(
                f"{str(idx).zfill(5)}", data=data
            )
        return data

    def reconstruct_noisy(self, o, f, data, idx):
        listGpuNames = gpu.getGpuNames()
        gpuids = gpu.getGpuIds(listGpuNames[0])

        print(data.shape)

        n_angles = np.random.randint(20, 120)  # RSD: How many projections?
        # n_angles = 500
        angles = np.linspace(0, 2 * np.pi, n_angles, endpoint=False)
        phantom_geo = PhantomGeometry()
        geo = phantom_geo()

        projs = tigre.Ax(
            data, geo, angles, gpuids=gpuids
        )  # RSD: Need shape (angles, height, width)

        # plt.imshow(projs[0], cmap="gray")
        # plt.show()

        # print(projs.shape)

        geo.rotDetector = np.array([0, 0, 0])
        # RSD: Reconstruction
        rec = algs.fdk(
            projs, geo, angles, gpuids=gpuids
        )  # RSD: Need same shape as target EQNR gave (height, width, width) Plus renormalization.
        rec = (rec - np.min(rec)) / (np.max(rec) - np.min(rec))

        o[ReconstructionsDataCT.NOISY_KEY].create_dataset(
            f"{str(idx).zfill(5)}", data=rec
        )

        return


class TomoBankDataCT(ReconstructionsDataCT):
    def __init__(self, root, name, o_root, o_name):
        super().__init__(o_root, o_name)
        self.root = root
        self.name = name

    def reconstruct_target(
        self, o, f, i, depth=512
    ):  # RSD: Update this due to information included in the files.

        data = np.squeeze(f[ReconstructionsDataCT.TOMOBANK_TARGET])
        data = np.broadcast_to(data, (data.shape[0], data.shape[1], depth))
        o[ReconstructionsDataCT.TARGET_KEY].create_dataset(
            f"{str(i).zfill(5)}", data=data
        )
        # sino = sino.transpose(2, 0, 1) #RSD: Consider, unsure on format for reconstruction
        return data

    def reconstruct_noisy(self, o, f, data, i, n_voxels=512):
        listGpuNames = gpu.getGpuNames()
        gpuids = gpu.getGpuIds(listGpuNames[0])

        n_angles = np.random.randint(45, 200)  # RSD: How many projections?
        angles = np.linspace(0, 2 * np.pi, n_angles, endpoint=False)
        geo = tigre.geometry(
            mode="cone", default=True, nVoxel=[data.shape[0], data.shape[1], n_voxels]
        )
        projs = tigre.Ax(data, geo, angles, gpuids=gpuids)

        # RSD: Add noise if needed. For now undersampling.

        # RSD: Reconstruction
        rec = algs.fdk(projs, geo, angles, gpuids=gpuids)
        o[ReconstructionsDataCT.NOISY_KEY].create_dataset(
            f"{str(i).zfill(5)}", data=rec
        )

        return


class EquinorDataCT(ReconstructionsDataCT):
    def __init__(self, root, name, o_root, o_name):
        super().__init__(o_root, o_name)
        self.root = root
        self.name = name

        return

    def reconstruct_target(self, o, f, idx, not_use_depth=256):
        listGpuNames = gpu.getGpuNames()
        gpuids = gpu.getGpuIds(listGpuNames[0])

        data = np.squeeze(f[ReconstructionsDataCT.EQNR_PROJECTIONS])
        with open(os.path.join(self.root, f"{self.name}.pkl"), "rb") as g:
            geo = pkl.load(g)
        # geo = open(pkl.load())  # f["geometry"]
        angles = np.array(f["angles"])  # geo["angles"]

        rec = algs.fdk(data, geo, angles, gpuids=gpuids)

        o[ReconstructionsDataCT.TARGET_KEY].create_dataset(
            f"{str(idx).zfill(5)}", data=rec
        )
        # sino = sino.transpose(2, 0, 1) #RSD: Consider, unsure on format for reconstruction
        return data

    def reconstruct_noisy(
        self, o, f, data, idx, dyn_slice=True
    ):  # RSD: Remember to implement dynamical slice
        listGpuNames = gpu.getGpuNames()
        gpuids = gpu.getGpuIds(listGpuNames[0])

        with open(os.path.join(self.root, f"{self.name}.pkl"), "rb") as g:
            geo = pkl.load(g)

        angles = np.array(
            f["angles"]
        )  # RSD: Constructed, not from file. Perhaps change this.
        data = np.squeeze(f[ReconstructionsDataCT.EQNR_PROJECTIONS])
        print(data.shape)

        # RSD: Add noise if needed. For now undersampling.

        n_projs = np.random.randint(
            len(angles) // 16, len(angles)
        )  # RSD: How many projections?
        print(len(angles), n_projs)
        slicing = len(angles) // n_projs
        if dyn_slice:
            angles = angles[:n_projs]
            data = data[:n_projs]
        else:
            angles = angles[::slicing]
            data = data[::slicing]

        rec = algs.fdk(data, geo, angles, gpuids=gpuids)

        o[ReconstructionsDataCT.NOISY_KEY].create_dataset(
            f"{str(idx).zfill(5)}", data=rec
        )
        return


class EquinorDynamicCT(EquinorDataCT):
    """
    Class to handle processed projections from 4DCT experiments at Equinor.
    The class can add reconstruction to machine learning dataset format.
    Additionally, it can reconstruct the data and plot the results, with the added option to merge datasets.
    """

    def __init__(self, root, name, o_root, o_name):
        super().__init__(self, root, name, o_root, o_name)
        return

    def reconstruct_noisy(self, o, f, data, idx, dyn_slice=True):
        listGpuNames = gpu.getGpuNames()
        gpuids = gpu.getGpuIds(listGpuNames[0])

        with open(os.path.join(self.root, f"{self.name}.pkl"), "rb") as g:
            geo = pkl.load(g)

        angles = np.array(f["angles"])
        # data = np.squeeze(f[ReconstructionsDataCT.EQNR_PROJECTIONS]) #RSD: More to it than that
        pass  # RSD: Not to be used yet.

    def reconstruct_singles(self):
        pass

    def plot_reconstruction(self, idx):
        pass

    def plot_4DCT(self):
        pass

    def merge_datasets(self):
        pass


class EquinorReconstructions(ReconstructionsDataCT):
    """
    Class to handle the Equinor reconstructions in .raw format, and reconstruct noisy samples.
    """

    def __init__(self, root, name, o_root, o_name):
        super().__init__(o_root, o_name)
        self.root = root
        self.name = name

        return

    def reconstruct_target(self, o, f, idx):

        dat_file = pd.read_csv(
            os.path.join(self.root, f"{self.name}.dat"),
            sep=":",
            header=None,
            names=["attribute", "value"],
            index_col=0,
        )
        resolution = dat_file.loc["resolution", "value"].split(" ")
        voxel_size = dat_file.loc["SliceThickness", "value"].split(" ")
        voxel_size = [float(i) for i in voxel_size]  # RSD: Use?
        w, h, d = [int(i) for i in resolution]

        path_raw = os.path.join(self.root, f"{self.name}.raw")
        with open(path_raw, "rb") as g:
            data = np.fromfile(g, dtype=np.float32).reshape(w, h, d)

        o[ReconstructionsDataCT.TARGET_KEY].create_dataset(
            f"{str(idx).zfill(5)}", data=data
        )
        return data

    def reconstruct_noisy(self, o, f, data, idx, n_angles, dyn_slice=False):

        listGpuNames = gpu.getGpuNames()
        gpuids = gpu.getGpuIds(listGpuNames[0])

        n_angles = n_angles  # np.random.randint(45, 200)  # RSD: How many projections?
        angles = np.linspace(0, 2 * np.pi, n_angles, endpoint=False)

        # RSD: This part is uncertain. Possibly choose between micro and industrial instead.
        with open(os.path.join(self.root, f"{self.name}.pkl"), "rb") as g:
            geo = pkl.load(g)

        projs = tigre.Ax(data, geo, angles, gpuids=gpuids)

        # RSD: Add noise if needed. For now undersampling.

        # RSD: Reconstruction. Choose algorithm.
        rec = algs.fdk(projs, geo, angles, gpuids=gpuids)
        o[ReconstructionsDataCT.NOISY_KEY].create_dataset(
            f"{str(idx).zfill(5)}", data=rec
        )
        return


def get_dataset_keys(f):
    keys = []
    f.visit(lambda key: keys.append(key) if isinstance(f[key], h5py.Dataset) else None)
    return keys


# # RSD: Reading raw file
# def read_raw_data(parent, filename, dat_parent, dat_filename):

#     # dat_file = pd.read_table(os.path.join(dat_parent, dat_filename))

#     # w, h, d = (
#     #     dat_file["width"][0],
#     #     dat_file["height"][0],
#     #     dat_file["depth"][0],
#     # )  # RSD: ISH
#     w, h, d = 512, 512, 512  # RSD: ISH

#     full = os.path.join(parent, filename)
#     with open(full, "rb") as f:
#         data = np.fromfile(f, dtype=np.float32)

#     # data = data.reshape((d, h, w))

#     return data


# def process_raw_file_reconstruction(parent, filenames, dat_parent, dat_filenames):
#     # RSD: Process raw files and save as h5 file.
#     # RSD: Need to figure out how to do reconstruction.

#     for i, (filename, dat) in enumerate(zip(filenames, dat_filenames)):
#         data = read_raw_data(parent, filename, dat_parent[i], dat_filenames[i])
#         # RSD: Back to sinogram?
#         # Save sinogram, and target as h5 file.
#         # RSD: Save as h5 file.
#     return


# # def merge_h5(parent, names, output, oname):
# #     """
# #     Takes prepared hdf5 files and merges them into one.
# #     """
# #     o = h5py.File(os.path.join(output, f"{oname}.h5"), "w")
# #     for i, name in enumerate(names):

# #     return


# def sinogram_2_datafile(
#     parent, names, output, oname, depth=256, keys=["exchange", "data"]
# ):  # RSD: Alternatively, produce 3D from a single reconstruction?
#     """
#     Takes TomoBank data, especially phantoms, and expand them in another dimension. save as h5 file.
#     Necessary with flatfield correction etc?
#     """

#     o = h5py.File(os.path.join(output, f"{oname}.h5"), "w")
#     o.create_group("sinogram3D")

#     for i, name in enumerate(names):
#         f = h5py.File(os.path.join(parent, f"{name}.h5"), "r")
#         sino = np.squeeze(f[keys[0]][keys[1]])
#         sino = np.broadcast_to(
#             sino, (sino.shape[0], sino.shape[1], depth)
#         )  # Leave depth to be the last dimension
#         f.close()

#         # RSD: Save as h5 file. Await the reconstruction step. Have to figure out how.

#         o["sinogram3D"].create_dataset(f"{i}", data=sino)  # Or use name?
#     o.close()
#     return


# def target_2_datafile(parent, names, output, depth=256, keys=["exchange", "gt"]):
#     """
#     Takes TomoBank data, especially phantoms, and expand the ground truth in another dimension. Writes to h5 file.
#     """
#     o = h5py.File(os.path.join(output, f"{name}3D.h5"), "w")
#     o.create_group("target3D")
#     for i, name in enumerate(names):

#         f = h5py.File(os.path.join(parent, f"{name}.h5"), "r")
#         target = np.squeeze(f[keys[0]][keys[1]])
#         target = np.broadcast_to(
#             target, (target.shape[0], target.shape[1], depth)
#         )  # Leave depth to be the last dimension
#         f.close()

#         # RSD: Save as h5 file.

#         o["target3D"].create_dataset(
#             f"{str(i).zfill(5)}", data=target
#         )  # Or use name? Check with datafile class.
#     o.close()
#     return


# def phantom_sinogram_reconstruction(
#     parent,
#     output,
#     num_projections,
#     dim_voxels=[512, 512, 512],
# ):
#     """
#     Perform reconstruction and save to h5 file.
#     Assumes h5 file with sinogram3D and target3D.
#     Reconstructs directly using the sinogram as projections.
#     Should be copied and be performed with more and fewer projections. Add on later.
#     Would simply be to repeat the process with different number of projections and somehow merge even though indexing becomes an issue.

#     Not sure if this works.
#     """
#     listGpuNames = gpu.getGpuNames()
#     gpuids = gpu.getGpuIds(listGpuNames[0])

#     o = h5py.File(os.path.join(parent, f"{output}.h5"), "w")
#     o.create_group("noisy3D")

#     f = h5py.File(os.path.join(parent, f"{output}.h5"), "r")

#     sino_geo = tigre.geometry(mode="cone", default=True, nVoxel=dim_voxels)
#     angles = np.linspace(0, 2 * np.pi, num_projections, endpoint=False)

#     for i, sinogram3D in enumerate(f["sinogram3D"]):

#         rec = algs.fdk(sinogram3D, sino_geo, angles, gpuids=gpuids)

#         o["noisy3D"].create_dataset(f"{i}", data=rec)

#     f.close()
#     o.close()

#     return


# def undersampled_reconstruction():
#     """
#     Figure out what to do with this thing. For now, one cannot do cropping on the fly while training. It would require to do the entire reconstruction on the fly.
#     Instead, choose random number of projections at (random angles)?, and reconstruct and save to hdf5 file.
#     Also, figure out the environments. Transfer CIL-environment to 4DCT since they are bound to work together. If not, some think will have to be conducted by each environment.
#     """
#     return


# def phantom_reconstruction(parent, output, n_projections, dim_voxels=[512, 512, 512]):
#     """
#     Reconstruction of phantoms.
#     Assumes h5 file with sinogram3D and target3D.
#     Skips the use of sinogram. Or is projections the same as sinogram? Should be.
#     """

#     listGpuNames = gpu.getGpuNames()
#     gpuids = gpu.getGpuIds(listGpuNames[0])

#     o = h5py.File(os.path.join(parent, f"{output}.h5"), "w")
#     o.create_group("noisy3D")

#     f = h5py.File(os.path.join(parent, f"{output}.h5"), "r")

#     sino_geo = tigre.geometry(mode="cone", default=True, nVoxel=dim_voxels)
#     angles = np.linspace(0, 2 * np.pi, n_projections, endpoint=False)

#     for i, target3D in enumerate(f["target3D"]):

#         projs = tigre.Ax(target3D, sino_geo, angles, gpuids=gpuids)

#         rec = algs.fdk(projs, sino_geo, angles, gpuids=gpuids)

#         o["noisy3D"].create_dataset(f"{i}", data=rec)

#     f.close()
#     o.close()

#     return


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
