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
import scipy.ndimage as nd


class ReconstructionsDataCT:
    NOISY_KEY = "noisy3D"
    TARGET_KEY = "target3D"
    ENHANCED_KEY = "enhanced3D"
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

        self.methods = {
            "fdk": algs.fdk,
            "sirt": algs.sirt,
            # "osem": algs.osem,
            # RSD: TODO: Add more methods
        }

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

    def add_item(self, obj, n_angles=100, method="fdk"):
        o = h5py.File(self.full_path, "r+")
        f = obj.read_item(obj)
        data = obj.reconstruct_target(o, f, self.__len__())
        obj.reconstruct_noisy(
            o, f, data, self.__len__(), n_angles=n_angles, method=method
        )
        f.close()
        o.close()
        return

    def read_item(self, obj):
        f = h5py.File(os.path.join(obj.root, f"{obj.name}.h5"), "r")
        return f

    def __len__(self):
        return len(h5py.File(self.full_path)[self.NOISY_KEY].keys())

    def process_data(self, objects: list, n_angles=100, method="fdk", undersampling_factor=True):
        o = h5py.File(self.full_path, "r+")

        for i, obj in enumerate(objects):
            index = self.__len__()
            # f = h5py.File(os.path.join(obj.root, f"{obj.name}.h5"), "r")
            f = obj.read_item(obj)
            data = obj.reconstruct_target(o, f, index)
            obj.reconstruct_noisy(o, f, data, index, n_angles=n_angles, method=method, undersampling_factor=undersampling_factor)
            # f.close() #RSD: Issue for EQNRRec not h5 file.

        o.close()
        return
    
    def copy_from_other(self, other):

        with h5py.File(self.full_path, "r+") as tofile:
            with h5py.File(other.full_path, "r") as fromfile:

                for j in tqdm.trange(other.__len__()):
                     idx = self.__len__()

                     tofile[ReconstructionsDataCT.TARGET_KEY].create_dataset(str(idx).zfill(5), data = fromfile[ReconstructionsDataCT.TARGET_KEY][str(j).zfill(5)])
                     tofile[ReconstructionsDataCT.NOISY_KEY].create_dataset(str(idx).zfill(5), data = fromfile[ReconstructionsDataCT.NOISY_KEY][str(j).zfill(5)])
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


class TigreGeometry:
    """
    Parent class for the different geometries used in TIGRE.
    """

    def __init__(self):
        self.geo = tigre.geometry(mode="cone", default=True)

        self.geo.DSD = self.values["DSD"]  # Distance Source Detector (mm)
        self.geo.DSO = self.values["DSO"]  # Distance Source Origin (mm)

        self.geo.nDetector = self.values["nDetector"]  # number of pixels (px)
        self.geo.dDetector = self.values["dDetector"]  # size of each pixel (mm)
        self.geo.sDetector = (
            self.geo.dDetector * self.geo.nDetector
        )  # total size of the detector (mm)

        self.geo.nVoxel = np.array(
            [self.geo.nDetector[0], self.geo.nDetector[1], self.geo.nDetector[1]]
        )

        magnification = lambda x: x * self.values["DSO"] / self.values["DSD"]
        self.values["dVoxel"] = np.array(
            [
                magnification(self.values["dDetector"][0]),
                magnification(self.values["dDetector"][1]),
                magnification(self.values["dDetector"][1]),
            ]
        )
        self.geo.dVoxel = self.values["dVoxel"]  # size of each voxel
        self.geo.sVoxel = self.geo.dVoxel * self.geo.nVoxel  # total size of the image

        self.geo.offOrigin = np.array([0, 0, 0])  # Offset of image from origin (mm)
        self.geo.offDetector = np.array([0, 0])  # Offset of Detector

        self.geo.accuracy = 0.5  # Accuracy of FWD proj    (vx/sample)

        self.geo.COR = 0

        self.geo.rotDetector = self.values["rotation"]  # Rotation of detector

    def set_COR(self, COR):
        self.geo.COR = COR
        return

    def __call__(self):
        return self.geo

    def set_roi(self, roi):
        """
        Possibly needed? But we also read from file?
        """
        self.geo.nDetector = np.array([roi[0], roi[1]])
        self.geo.sDetector = self.geo.dDetector * self.geo.nDetector
        self.geo.nVoxel = np.array([roi[0], roi[1], roi[1]])
        self.geo.sVoxel = self.geo.dVoxel * self.geo.nVoxel
        return


class IndustrialGeometry(TigreGeometry):
    """
    Create Industrial Geometry used at Industrial-CT Instrument.
    """

    def __init__(self):
        self.values = {
            "DSD": 1350,  # Distance Source Detector (mm)
            "DSO": 930,  # Distance Source Origin (mm)
            "nDetector": np.array([2048, 2048]),  # number of pixels (px)
            "dDetector": np.array([0.200, 0.200]),  # size of each pixel (mm)
            "dVoxel": np.array([0.125, 0.125, 0.125]),  # Effective voxel pitch (mm)
            "rotation": np.array([0, 0, 0]),  # Rotation of detector
        }

        super().__init__()
        return


class MicroGeometry(TigreGeometry):
    """
    Create Micro Geometry used at Micro-CT Instrument. Issue is that the standard setting uses zoom.
    """

    def __init__(self):
        self.values = {
            "DSD": 502,  # 72  # Distance Source Detector (mm)
            "DSO": 47,  # Distance Source Origin (mm)
            "nDetector": np.array([1920, 1536]),  # number of pixels (px)
            "dDetector": np.array([0.127, 0.127]),  # size of each pixel (mm)
            "dVoxel": np.array(
                [0.01189, 0.01189, 0.01189]
            ),  # Effective voxel pitch (mm)
            "rotation": np.array([0, 0, 0]),  # Rotation of detector
        }

        super().__init__()

        return


class PhantomGeometry(TigreGeometry):
    """
    Geometry associated with the phantom.
    (This should be obsolete, and instead use the MicroGeometry or the IndustrialGeometry)
    Interesting to note different artefacts given geometry.
    """

    def __init__(self, default=True):
        if default:
            self.values = {
                "DSD": 1350,  # Distance Source Detector (mm)
                "DSO": 930,  # Distance Source Origin (mm)
                "nDetector": np.array([1024, 512]),  # number of pixels (px)
                "dDetector": np.array([0.2, 0.2]),  # size of each pixel (mm)
                "dVoxel": np.array([0.2, 0.2, 0.2]),  # Effective voxel pitch (mm)
                "rotation": np.array([0, 0, 0]),  # Rotation of detector
            }

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

        self.geo = phan_geo

        return


class TomoBankPhantomCT(ReconstructionsDataCT):
    """
    Creates a dataset for a TomoBank phantom.
    """

    def __init__(self, root, name, o_root, o_name, geo=PhantomGeometry()):
        super().__init__(o_root, o_name)
        self.root = root
        self.name = name
        self.geo = geo

    def reconstruct_target(self, o, f, idx, depth=512, sino=False):
        if sino:  # RSD: Ignored for now. Not much to gain.
            angles = np.squeeze(
                np.array(f[ReconstructionsDataCT.TOMOBANK_ANGLES])
            )  # np.squeeze(f[ReconstructionsDataCT.EQNR_ANGLES])
            data = np.squeeze(f[ReconstructionsDataCT.TOMOBANK_SINO])
            o[ReconstructionsDataCT.SINO_KEY].create_dataset(
                f"{str(idx).zfill(5)}", data=data
            )

            try:
                isinstance(o[ReconstructionsDataCT.NOISY_KEY], h5py.Group)
            except:
                o.create_group(ReconstructionsDataCT.NOISY_KEY)
            finally:
                o[ReconstructionsDataCT.EQNR_ANGLES].create_dataset(
                    f"{str(idx).zfill(5)}", data=angles
                )
        else:
            data = np.squeeze(f[ReconstructionsDataCT.TOMOBANK_TARGET]).astype(
                np.float32
            )[np.newaxis, :, :]

            data_layer = np.vstack([data.copy()] * 32).astype(np.float32)
            blancks = np.zeros((32, data.shape[1], data.shape[2])).astype(np.float32)
            for k in range(depth // 32):
                if k == 0:
                    data = data_layer
                    continue
                elif k % 2 == 0:
                    data = np.vstack((data, nd.rotate(data_layer, angle=(k+1)*180/(depth//32), axes=(2,1), reshape=False)))
                    # RSD: TODO: Add some rotation to make it less symmetric. Check if nice. 
                else:
                    data = np.vstack((data, blancks))

            o[ReconstructionsDataCT.TARGET_KEY].create_dataset(
                f"{str(idx).zfill(5)}", data=data
            )
        return data

    def reconstruct_noisy(self, o, f, data, idx, n_angles, method="fdk", undersampling_factor=True):
        listGpuNames = gpu.getGpuNames()
        gpuids = gpu.getGpuIds(listGpuNames[0])

        if undersampling_factor:
            n_angles = int(np.pi/2*data.shape[-1]//n_angles)
        else:
            n_angles = n_angles

        angles = np.linspace(0, 2 * np.pi, n_angles, endpoint=False)
        # phantom_geo = PhantomGeometry()
        # geo = self.geo()  # phantom_geo()
        geo = self.set_geo_dim(data.shape[0], data.shape[-1])

        projs = tigre.Ax(data, geo, angles, gpuids=gpuids)

        geo.rotDetector = np.array([0, 0, 0])
        # RSD: Reconstruction
        # rec = algs.fdk(projs, geo, angles, gpuids=gpuids)
        rec = self.methods[method](projs, geo, angles, gpuids=gpuids)
        rec = (rec - np.min(rec)) / (np.max(rec) - np.min(rec))

        o[ReconstructionsDataCT.NOISY_KEY].create_dataset(
            f"{str(idx).zfill(5)}", data=rec
        )

        return
    
    def set_geo_dim(self, height, width):
        '''
        Calls the function to retrieve the TIGRE GEOMETRY object, and ensures the expected shapes are correct.
        '''
        geo = self.geo()
        geo.nDetector = np.array([height, width])
        geo.nVoxel = np.array([height, width, width])
        geo.sDetector = geo.nDetector * geo.dDetector
        geo.sVoxel = geo.nVoxel * geo.dVoxel
        return geo



class TomoBankDataCT(ReconstructionsDataCT):
    """
    Creating dataset for experimental TomoBank data. (Not up to date)
    """

    def __init__(self, root, name, o_root, o_name):
        super().__init__(o_root, o_name)
        self.root = root
        self.name = name

    def reconstruct_target(self, o, f, i, depth=512):
        """
        Reconstructs the target from the hdf5 file and saves it to the output file.
        Update if need for experimental data online (More data, different format)
        """
        data = np.squeeze(f[ReconstructionsDataCT.TOMOBANK_TARGET])
        data = np.broadcast_to(data, (data.shape[0], data.shape[1], depth))
        o[ReconstructionsDataCT.TARGET_KEY].create_dataset(
            f"{str(i).zfill(5)}", data=data
        )
        return data

    def reconstruct_noisy(self, o, f, data, i, n_angles, n_voxels=512):
        listGpuNames = gpu.getGpuNames()
        gpuids = gpu.getGpuIds(listGpuNames[0])

        angles = np.linspace(0, 2 * np.pi, n_angles, endpoint=False)
        geo = tigre.geometry(
            mode="cone", default=True, nVoxel=[data.shape[0], data.shape[1], n_voxels]
        )
        projs = tigre.Ax(data, geo, angles, gpuids=gpuids)

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
        angles = np.array(f["angles"])

        rec = algs.fdk(
            data, geo, angles, gpuids=gpuids
        )  # RSD: TODO: Implement method for choosing reconstruction

        o[ReconstructionsDataCT.TARGET_KEY].create_dataset(
            f"{str(idx).zfill(5)}", data=rec
        )
        return data

    def reconstruct_noisy(
        self, o, f, data, idx, n_angles, wedge_slice=False, method="fdk"
    ):  # RSD: Remember to implement dynamical slice
        listGpuNames = gpu.getGpuNames()
        gpuids = gpu.getGpuIds(listGpuNames[0])

        with open(os.path.join(self.root, f"{self.name}.pkl"), "rb") as g:
            geo = pkl.load(g)

        angles = np.array(f["angles"])
        data = np.squeeze(f[ReconstructionsDataCT.EQNR_PROJECTIONS])

        slicing = len(angles) // n_angles
        if wedge_slice:
            angles = angles[:n_angles]
            data = data[:n_angles]
        else:
            angles = angles[::slicing]
            data = data[::slicing]

        # rec = algs.fdk(data, geo, angles, gpuids=gpuids)
        rec = self.methods[method](data, geo, angles, gpuids=gpuids)

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
        super().__init__(root, name, o_root, o_name)

        return

    def reconstruct_group(self, data, angles, geo, method="fdk"):
        listGpuNames = gpu.getGpuNames()
        gpuids = gpu.getGpuIds(listGpuNames[0])

        rec = self.methods[method](data, geo, angles, gpuids=gpuids)
        return rec

    def reconstruct_idx(self, idx, method="fdk"):
        with h5py.File(os.path.join(self.root, f"{self.name}.h5"), "r") as f:
            data = np.squeeze(
                f[ReconstructionsDataCT.EQNR_PROJECTIONS][str(idx).zfill(5)]
            )
            angles = np.squeeze(f[ReconstructionsDataCT.EQNR_ANGLES][str(idx).zfill(5)])

        with open(os.path.join(self.root, f"{self.name}.pkl"), "rb") as g:
            geo = pkl.load(g)

        rec = self.reconstruct_group(data, angles, geo, method=method)
        return rec

    def make_projection_group(self, idx, fibonacci):
        with h5py.File(os.path.join(self.root, f"{self.name}.h5"), "r") as f:
            data = np.squeeze(
                f[ReconstructionsDataCT.EQNR_PROJECTIONS][str(idx).zfill(5)]
            )
            angles = np.squeeze(f[ReconstructionsDataCT.EQNR_ANGLES][str(idx).zfill(5)])

            for elem in range(1, fibonacci):
                new_data = np.squeeze(
                    f[ReconstructionsDataCT.EQNR_PROJECTIONS][str(idx + elem).zfill(5)]
                )
                new_angles = np.squeeze(
                    f[ReconstructionsDataCT.EQNR_ANGLES][str(idx + elem).zfill(5)]
                )
                data = np.concatenate((data, new_data), axis=0)
                angles = np.concatenate(
                    (angles, new_angles), axis=0
                )  # RSD TODO: Check if equivalent to np.stack

        with open(os.path.join(self.root, f"{self.name}.pkl"), "rb") as g:
            geo = pkl.load(g)

        return data, angles, geo

    def save_rec_timestamp(self, rec):
        pass

    def reconstruct_singles(self, method="fdk"):
        timestamps = len(
            h5py.File(os.path.join(self.root, f"{self.name}.h5"))[
                self.EQNR_ANGLES
            ].keys()
        )
        # RSD: Assumes that the group is empty. Reasonable assumption.
        for idx in range(tqdm.tqdm(timestamps)):
            rec = self.reconstruct_idx(idx, method=method)
            self.add_undersampled_rec(rec, idx)

    def add_undersampled_rec(self, rec, idx):
        with h5py.File(os.path.join(self.o_root, f"{self.o_name}.h5"), "a") as o:
            o[ReconstructionsDataCT.NOISY_KEY].create_dataset(
                f"{str(idx).zfill(5)}", data=rec
            )
        return

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
                    data = data.reshape((d, w, h))
            except:
                with open(path_raw, "rb") as g:
                    data = np.fromfile(g, dtype="uint8")  # .reshape((w, h, d))
                    data = data.reshape((d, w, h))

        except:
            try:
                import nsiefx

                data = np.zeros((d, w, h), dtype=np.float32)
                # RSD: Believe crossection indexed by first index.
                with nsiefx.open(
                    os.path.join(self.root, f"{self.name}.nsihdr")
                ) as volume:
                    for slice_idx in range(d):
                        data[slice_idx] = volume.read_slice(slice_idx)

            except:
                KeyError("Did not find dataset, or unexpected format.")

        finally:
            data = data.astype(np.float32)
            data = (data - np.min(data)) / (np.max(data) - np.min(data))
            o[ReconstructionsDataCT.TARGET_KEY].create_dataset(
                f"{str(idx).zfill(5)}", data=data
            )
            return data

    def reconstruct_noisy(self, o, f, data, idx, n_angles, method="fdk", undersampling_factor=True):
        listGpuNames = gpu.getGpuNames()
        gpuids = gpu.getGpuIds(listGpuNames[0])

        if undersampling_factor:
            n_angles = int(np.pi/2*data.shape[-1]//n_angles)
        else:
            n_angles = n_angles  # np.random.randint(45, 200)  # RSD: How many projections?

        angles = np.linspace(0, 2 * np.pi, n_angles, endpoint=False) 
        #RSD: TODO: Should be distributed as Golden Ratio?

        default_geo = MicroGeometry()
        geo = default_geo()

        geo.nVoxel = self.dims
        geo.nDetector = np.array([self.dims[0], self.dims[1]])
        geo.sVoxel = geo.nVoxel * geo.dVoxel
        geo.sDetector = geo.nDetector * geo.dDetector

        projs = tigre.Ax(data, geo, angles, gpuids=gpuids)
        plt.imshow(projs[0], cmap="gray")
        plt.show()

        rec = self.methods[method](projs, geo, angles, gpuids=gpuids)
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
