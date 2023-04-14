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

    def __init__(self, data_root, data_name, n_angles=100):
        self.data_root = data_root
        self.data_name = data_name
        self.full_path = os.path.join(self.data_root, f"{self.data_name}.h5")
        self.n_angles = n_angles  # RSD: Redo

        # try:
        #     o = h5py.File(self.full_path, "w-")
        # except:
        #     o = h5py.File(self.full_path, "r+")
        o = h5py.File(self.full_path, "a")

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
        f = obj.read_item(
            obj
        )  # h5py.File(os.path.join(obj.root, f"{obj.name}.h5"), "r")
        data = obj.reconstruct_target(o, f, self.__len__())
        obj.reconstruct_noisy(o, f, data, self.__len__(), self.n_angles)
        f.close()
        o.close()
        return

    def read_item(self, obj):
        f = h5py.File(os.path.join(obj.root, f"{obj.name}.h5"), "r")
        return f

    def __len__(self):
        return len(h5py.File(self.full_path)[self.NOISY_KEY].keys())

    def process_data(self, objects: list):

        o = h5py.File(self.full_path, "r+")

        for i, obj in enumerate(objects):

            index = self.__len__()
            # f = h5py.File(os.path.join(obj.root, f"{obj.name}.h5"), "r")
            f = obj.read_item(obj)
            data = obj.reconstruct_target(o, f, index)
            obj.reconstruct_noisy(o, f, data, index, self.n_angles) # Close here instead. Fix
            # f.close() #RSD: Issue for EQNRRec not h5 file.

        o.close()
        return

    def visualise(self, idx: list = [-1]):

        o = h5py.File(self.full_path, "r")
        fig, ax = plt.subplots(1, 2)

        for i in idx:
            if i == -1:
                i = self.__len__() - 1
            midsection = o[self.TARGET_KEY][f"{str(i).zfill(5)}"].shape[0] // 2
            ax[0].imshow(
                o[self.TARGET_KEY][f"{str(i).zfill(5)}"][midsection], cmap="Greys"
            )
            ax[1].imshow(
                o[self.NOISY_KEY][f"{str(i).zfill(5)}"][midsection], cmap="Greys"
            )
            ax[0].set_axis_off()
            ax[1].set_axis_off()
            plt.show()
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

            # data = np.vstack([data] * depth)
            # Hardcoded, but does not really matter.
            data_layer = np.vstack([data.copy()] * 32).astype(np.float32)
            blancks = np.zeros((32, data.shape[1], data.shape[2])).astype(np.float32)
            for k in range(depth // 32):
                if k == 0:
                    data = data_layer
                    continue
                elif k % 2 == 0:
                    data = np.vstack((data, data_layer))
                else:
                    data = np.vstack((data, blancks))

            o[ReconstructionsDataCT.TARGET_KEY].create_dataset(
                f"{str(idx).zfill(5)}", data=data
            )
        return data

    def reconstruct_noisy(self, o, f, data, idx, n_angles):
        listGpuNames = gpu.getGpuNames()
        gpuids = gpu.getGpuIds(listGpuNames[0])

        print(data.shape)

        # n_angles = np.random.randint(20, 120)  # RSD: How many projections?
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

    def reconstruct_noisy(self, o, f, data, i, n_angles, n_voxels=512):
        listGpuNames = gpu.getGpuNames()
        gpuids = gpu.getGpuIds(listGpuNames[0])

        # n_angles = np.random.randint(45, 200)  # RSD: How many projections?
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
        self, o, f, data, idx, n_angles, dyn_slice=True
    ):  # RSD: Remember to implement dynamical slice
        listGpuNames = gpu.getGpuNames()
        gpuids = gpu.getGpuIds(listGpuNames[0])

        with open(os.path.join(self.root, f"{self.name}.pkl"), "rb") as g:
            geo = pkl.load(g)

        angles = np.array(f["angles"])
        data = np.squeeze(f[ReconstructionsDataCT.EQNR_PROJECTIONS])
        print(data.shape)

        # RSD: Add noise if needed. For now undersampling.

        # n_projs = np.random.randint(
        #     len(angles) // 16, len(angles)
        # )  # RSD: How many projections?
        # print(len(angles), n_projs)
        slicing = len(angles) // n_angles
        if dyn_slice:
            angles = angles[:n_angles]
            data = data[:n_angles]
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

    def reconstruct_noisy(self, o, f, data, idx, n_angles, dyn_slice=True):
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
        super().__init__(o_root, o_name)  # RSD: not really necessary.
        self.root = root
        self.name = name

        return

    def read_item(self, obj):
        return None  # Not used for this instance. Complicated, but will probably work.

    def reconstruct_target(self, o, f, idx):

        try:

            dat_file = pd.read_csv(
                os.path.join(self.root, f"{self.name}.dat"),
                sep=":",
                header=None,
                names=["attribute", "value"],
                index_col=0,
            )
            resolution = dat_file.loc["Resolution", "value"].split(" ")
            voxel_size = list(
                filter(None, dat_file.loc["SliceThickness", "value"].split(" "))
            )
            voxel_size = [float(i) for i in voxel_size]  # RSD: Use?
            res = list(filter(None, [i for i in resolution]))
            w, h, d = [int(i) for i in res]
            self.dims = np.array([d, w, h])

            path_raw = os.path.join(self.root, f"{self.name}.raw")
            try:
                with open(path_raw, "rb") as g:
                    data = np.fromfile(g, dtype="uint16")  # .reshape((w, h, d))
                    data = data.reshape((d,w,h))
            except:
                with open(path_raw, "rb") as g:
                    data = np.fromfile(g, dtype="uint8")  # .reshape((w, h, d))
                    data = data.reshape((d,w,h))

        except:
            try:
                import nsiefx

                data = np.zeros((d, w, h), dtype=np.float32)
                # RSD: Believe crossection indexed by first index.
                with nsiefx.open(os.path.join(self.root, f"{self.name}.nsihdr")) as volume:
                    for slice_idx in range(d):
                        data[slice_idx] = volume.read_slice(slice_idx)

            
            except: KeyError("Did not find dataset, or unexpected format.")
        
        finally:
        
            o[ReconstructionsDataCT.TARGET_KEY].create_dataset(
                        f"{str(idx).zfill(5)}", data=data.astype(np.float32)
                    )
            return data.astype(np.float32)           





    def reconstruct_noisy(self, o, f, data, idx, n_angles, dyn_slice=False):

        listGpuNames = gpu.getGpuNames()
        gpuids = gpu.getGpuIds(listGpuNames[0])

        n_angles = n_angles  # np.random.randint(45, 200)  # RSD: How many projections?
        angles = np.linspace(0, 2 * np.pi, n_angles, endpoint=False)

        # RSD: This part is uncertain. Possibly choose between micro and industrial instead.
        # Task is to create geo from reconstruction nsipro-file.
        # with open(os.path.join(self.root, f"{self.name}.pkl"), "rb") as g:
        #     geo = pkl.load(g)
        default_geo = PhantomGeometry() #RSD: TEST CASE IMPROVE
        geo = default_geo()

        # geo.nVoxel = np.array([1484, 1484, 1807])
        geo.nVoxel = self.dims
        geo.sVoxel = geo.nVoxel * geo.dVoxel

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
