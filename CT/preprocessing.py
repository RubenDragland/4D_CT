import tigre
import numpy as np
import scipy.ndimage as nd
from PIL import Image
import os
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import h5py
import pickle as pkl
import os
import re
import tqdm

"""
Here, we will handle tif files obtained at EQNR.
Perhaps make it as a class?
"""

# RSD: Make class to automate this process in case it has changed between scans.


class IndustrialGeometryEQNR:
    def __init__(
        self, default=True, nsprg=True, **kwargs
    ):  # RSD: Both nsprg and nspro exist.

        if default:

            self.values = {
                "DSD": 1350,  # Distance Source Detector (mm)
                "DSO": 930,  # Distance Source Origin (mm)
                "nDetector": np.array([2048, 2048]),  # number of pixels (px)
                "dDetector": np.array([0.2, 0.2]),  # size of each pixel (mm)
                "rotation": 0,
            }
        else:
            # try:
            self.read_from_file(kwargs["path"])
            # except:
            # raise ValueError("Please provide a path to the file.")

        golden_geometry = tigre.geometry(mode="cone", default=True)

        golden_geometry.DSD = self.values["DSD"]  # Distance Source Detector (mm)
        golden_geometry.DSO = self.values["DSO"]  # Distance Source Origin (mm)

        golden_geometry.nDetector = self.values["nDetector"]  # number of pixels (px)
        golden_geometry.dDetector = self.values["dDetector"]  # size of each pixel (mm)
        golden_geometry.sDetector = (
            golden_geometry.dDetector * golden_geometry.nDetector
        )  # total size of the detector (mm)

        golden_geometry.nVoxel = np.repeat(
            golden_geometry.nDetector[0], 3
        )  # number of voxels
        golden_geometry.dVoxel = np.repeat(
            golden_geometry.dDetector[0], 3
        )  # size of each voxel
        golden_geometry.sVoxel = (
            golden_geometry.dVoxel * golden_geometry.nVoxel
        )  # total size of the image

        golden_geometry.offOrigin = np.array(
            [0, 0, 0]
        )  # Offset of image from origin (mm)
        golden_geometry.offDetector = np.array([0, 0])  # Offset of Detector

        golden_geometry.accuracy = 0.5  # Accuracy of FWD proj    (vx/sample)

        golden_geometry.COR = 0

        golden_geometry.rotDetector = np.array([0, 0, 0])  # Rotation of detector
        # golden_geometry.rotDetector = np.array([self.values["rotation"], 0, 0])  # Rotation of detector

        self.golden_geometry = golden_geometry
        self.values = self.values
        return  # (golden_geometry, self.values)

    def read_dict_from_file(self, path):

        with open(path, "r") as f:
            data = f.readlines()
            data = [line.strip() for line in data]
            tot_dict = {}
            keys = []
            for line in data:
                k_start = line.find("<")
                k_end = line.find(">")
                key = line[k_start + 1 : k_end]
                keys.append(key)
                if f"{key[1:]}" in keys[:-1]:
                    index = keys[:-1].index(key[1:])
                    keys = keys[:index]
                    continue
                info = line[k_end + 1 :]
                if len(info) == 0:
                    continue
                else:
                    tot_dict["/".join(keys)] = info
                    keys.pop()
        return tot_dict

    def fix_rotation(self, file_info, attributes):
        rotation_raw = file_info[attributes["rotation"]]
        rotation = re.findall(r"\d+", rotation_raw)
        assert len(rotation) == 1
        rotation = float(rotation[0].replace(",", "."))
        print(f"Rotation: {rotation}")
        return rotation

    def read_from_file(self, path, config="CT"):
        # RSD: Read from file.

        root = f"North Star Imaging DR-Program 1.0/step/{config} Project Configuration/Technique Configuration/"

        attributes = {
            "DSD": f"{root}Setup/source to detector distance",  # Distance Source Detector (mm)
            "DSO": f"{root}Setup/source to table distance",  # Distance Source Origin (mm)
            "pixel_width": f"{root}Detector/pixel width microns",
            "pixel_height": f"{root}Detector/pixel height microns",
            "detector_pixel_width": f"{root}Detector/width pixels",
            "detector_pixel_height": f"{root}Detector/height pixels",
            "rotation": f"{root}Detector/rotation description",
        }

        file_info = self.read_dict_from_file(path)

        rotation = self.fix_rotation(file_info, attributes)

        self.values = {
            "DSD": float(file_info[attributes["DSD"]]),
            "DSO": float(file_info[attributes["DSO"]]),
            "nDetector": np.array(
                [
                    int(file_info[attributes["detector_pixel_width"]]),
                    int(file_info[attributes["detector_pixel_height"]]),
                ]
            ),  # number of pixels (px)
            "dDetector": np.array(
                [
                    float(file_info[attributes["pixel_width"]]) / 1000,
                    float(file_info[attributes["pixel_height"]]) / 1000,
                ]
            ),  # size of each pixel (mm)
            "rotation": rotation,
        }
        return


class ProjectionsEQNR:
    """
    A class to preprocess and centralise an ordinary CT measurement

    Attributes
    ----------

    root: str
        parent path to the data folder
    exp_name: str/list
        experiment name (Fix)
    o_root: str
        path to output data location
    number_of_projections: int
        Number of projections
    correction_parent: str
        Path to correction files
    name_flat: str
        Name of flat field correction file
    name_dark: str
        Name of dark field correction file
    geometry: str
        Path to .pkl with Tigre geometry corresponding to CT scanner geometry.
    roi: tuple
        Tuple with width and height of region of interest. Will crop middle section.
    rotation: int
        Rotation of projections.
    """

    def __init__(
        self,
        root,
        exp_name,
        o_root,
        number_of_projections,
        correction_parent=None,
        name_flat="gain0.tif",
        name_dark="offset.tif",
        geometry=None,
        roi=None,
        metallic_mean_n=0,  # RSD: Calculates angles instead of reading from file.
        rotation=0,
        mod=False,  #  RSD: Not yet implemented
    ):
        self.root = root
        self.exp_name = exp_name
        self.o_root = o_root
        self.number_of_projections = number_of_projections
        self.p_roots = [
            os.path.join(root, f"{exp_name}{str(i).zfill(5)}.tif")
            for i in range(number_of_projections)
        ]

        self.find_corrections(correction_parent)
        self.name_flat = name_flat
        self.name_dark = name_dark

        if roi is None:
            self.roi = self.load_tif(
                os.path.join(self.correction_parent, name_flat)
            ).shape
        else:
            self.roi = roi

        if geometry is None:
            geom_obj = IndustrialGeometryEQNR()
            self.geometry, geom_values = geom_obj.golden_geometry, geom_obj.values
        else:
            geom_obj = IndustrialGeometryEQNR(default=False, path=geometry)
            self.geometry, geom_values = geom_obj.golden_geometry, geom_obj.values
            self.geometry.nDetector = np.array([self.roi[0], self.roi[1]])
            self.geometry.nVoxel = np.array([self.roi[0], self.roi[0], self.roi[1]])
            self.geometry.sVoxel = self.geometry.dVoxel * self.geometry.nVoxel
            self.geometry.sDetector = self.geometry.dDetector * self.geometry.nDetector

        self.rotation = geom_values["rotation"] if rotation == 0 else rotation

        self.metallic_mean_n = metallic_mean_n
        if self.metallic_mean_n:
            angle_step = (
                180 * (self.metallic_mean_n + np.sqrt(4 + self.metallic_mean_n**2))
            ) % 360
            self.angles = np.arange(
                0, angle_step * self.number_of_projections, angle_step
            )
        else:
            self.angles = np.linspace(0, 2 * np.pi, self.number_of_projections)
            # self.angles = self.read_angles(os.path.join(self.root, f"{exp_name}"))
            # RSD: Check if ok.

        self.projections = None

    def __call__(self):

        self.projections = np.zeros(
            (self.number_of_projections, self.roi[0], self.roi[1])
        )

        for i, p_root in enumerate(self.p_roots):
            im = self.load_tif(p_root)

            im = self.normalise_projection(im)
            im = self.remove_defects(im)
            im = self.rotate_projection(im)
            im = self.crop_roi(im, self.roi)
            self.projections[i] = im

        self.save_h5()
        # self.plot_projection(0)

    def find_corrections(self, correction_parent):
        """Finds correction files in given folder"""
        self.correction_parent = (
            Path(self.root).parent if correction_parent == None else correction_parent
        )
        return

    def normalise_projection(self, proj: np.ndarray, tol=1e-6):
        """Performs flatfield and darkfield correction on projection image. Also, inverts images"""
        flat = self.load_tif(os.path.join(self.correction_parent, self.name_flat))
        dark = self.load_tif(os.path.join(self.correction_parent, self.name_dark))
        proj = (proj - dark) / (tol + flat - dark)
        proj = 1 - np.clip(proj, 0, 1)
        return proj

    def remove_defects(self, proj):
        """Remove defects from projection image. Applies median filter with 3x3 kernel"""
        defect_indices = pd.read_csv(
            os.path.join(self.correction_parent, "defective_pixels.defect"),
            skiprows=6,
            skipfooter=2,
            engine="python",
            dtype=int,
            sep=" ",
        ).to_numpy()

        for i, (s_index, f_index, one) in enumerate(defect_indices):

            kernel = proj[f_index - 1 : f_index + 2, s_index - 1 : s_index + 2]
            proj[f_index, s_index] = nd.median_filter(kernel, size=(3, 3))[1, 1]

        return proj

    def crop_roi(self, proj, roi):
        """
        Roi if width and height of the region of interest.
        Keeping center of rotation intact
        """
        shape = proj.shape
        slice_x = slice(shape[0] // 2 - roi[0] // 2, shape[0] // 2 + roi[0] // 2)
        slice_y = slice(shape[1] // 2 - roi[1] // 2, shape[1] // 2 + roi[1] // 2)
        return proj[slice_x, slice_y]

    def rotate_projection(self, proj):
        """
        Rotating projection according to nsipro angle.
        """
        return nd.rotate(proj, self.rotation, reshape=False)

    def load_tif(self, full_path):
        """Load tiff file using PIL, and return as numpy array"""
        im = Image.open(full_path)
        im = np.asarray(im)
        return im

    def read_angles(self, folder_path):
        full_path = os.path.join(folder_path, "positions.txt")
        raw_data = pd.read_csv(
            full_path, sep=" ", header=None, skipfooter=2, skiprows=1, engine="python"
        )
        angles = np.array(
            [float(str(angle).replace(",", ".")) for angle in raw_data[0]]
        )

        assert angles.dtype == np.float64
        "Wrong conversion. See read_angles(self)"
        return angles

    def save_h5(self):
        """Saving projections as h5 file with geometry data for reconstruction"""
        geom_root = os.path.join(self.o_root, f"{self.exp_name}.pkl")
        with open(geom_root, "wb+") as g:
            pkl.dump(self.geometry, g)
        with h5py.File(os.path.join(self.o_root, f"{self.exp_name}.h5"), "w") as f:
            f.create_dataset("projections", data=self.projections)
            f.create_dataset("angles", data=self.angles)
            # f.create_dataset("geometry", data=self.geometry)
            f.create_group("meta")
            f["meta"].attrs["metallic_mean_n"] = self.metallic_mean_n
            f["meta"].attrs["rotation"] = self.rotation
            f["meta"].attrs["roi"] = self.roi
            f["meta"].attrs["root"] = self.root
            f["meta"].attrs["correction_parent"] = self.correction_parent
            f["meta"].attrs["name_flat"] = self.name_flat
            f["meta"].attrs["name_dark"] = self.name_dark
            f["meta"].attrs["number_of_projections"] = self.number_of_projections
            f["meta"].attrs["exp_name"] = self.exp_name
            f["meta"].attrs["geometry"] = geom_root

        return

    def plot_projection(self, index=0):
        """Plot projection at index"""
        plt.imshow(self.projections[index], cmap="gray")
        plt.colorbar()
        plt.show()
        return

    def visualise(self, idx=0):
        """Visualise processed projection"""

        with h5py.File(os.path.join(self.o_root, f"{self.exp_name}.h5"), "r") as f:

            data = np.squeeze(np.array(f["projections"][f"{str(idx).zfill(5)}"]))

            plt.imshow(
                data,
                cmap="gray",
            )
            plt.colorbar()
            plt.show()
        return


class DynamicProjectionsEQNR(ProjectionsEQNR):
    """
    A class to preprocess and centralise a dynamical CT measurement

    Attributes
    ----------

    root: str
        parent path to the data folder
    exp_name: str/list
        experiment name (Fix)
    o_root: str
        path to output data location
    number_of_projections: int
        Number of projections in one revolution.
    correction_parent: str
        Path to correction files
    name_flat: str
        Name of flat field correction file
    name_dark: str
        Name of dark field correction file
    geometry: str
        Path to .nsprg with Tigre geometry corresponding to CT scanner geometry.
    roi: tuple
        Tuple with width and height of region of interest. Will crop middle section.
    rotation: int
        Rotation of projections.
    """

    def __init__(
        self,
        root,
        exp_name,
        o_root,
        number_of_projections,
        correction_parent=None,
        name_flat="gain0.tif",
        name_dark="offset.tif",
        geometry=None,
        roi=None,
        rotation=0,
    ):

        super().__init__(
            root,
            exp_name,
            o_root,
            number_of_projections,
            correction_parent=correction_parent,
            name_flat=name_flat,
            name_dark=name_dark,
            geometry=geometry,
            roi=roi,
            rotation=rotation,
        )
        # RSD: parent does some unnecessary work. Replace the things that are wrong.

        self.nproj_360 = number_of_projections
        self.nrevs = 20  # RSD: Change
        self.tot_steps = self.nproj_360 * self.nrevs

        self.revolution_folders = os.listdir(root)
        self.revolution_folders = [
            item
            for item in self.revolution_folders
            if item.startswith("Radiographs-step")
        ]

        self.revolution_folders = [
            (int((re.search(" \d+-", dir).group(0)).strip("-")), dir)
            for dir in self.revolution_folders
        ]
        self.revolution_folders = sorted(self.revolution_folders)

        return

    def init_save_h5(self):

        geom_root = os.path.join(self.o_root, f"{self.exp_name}.pkl")
        with open(geom_root, "wb+") as g:
            pkl.dump(self.geometry, g)

        with h5py.File(os.path.join(self.o_root, f"{self.exp_name}.h5"), "w") as f:
            f.create_group("meta")
            f["meta"].attrs["metallic_mean_n"] = self.metallic_mean_n
            f["meta"].attrs["rotation"] = self.rotation
            f["meta"].attrs["roi"] = self.roi
            f["meta"].attrs["root"] = self.root
            f["meta"].attrs["correction_parent"] = self.correction_parent
            f["meta"].attrs["name_flat"] = self.name_flat
            f["meta"].attrs["name_dark"] = self.name_dark
            f["meta"].attrs["number_of_projections"] = self.tot_steps
            f["meta"].attrs["nproj_360"] = self.nproj_360
            f["meta"].attrs["nrevs"] = self.nrevs
            f["meta"].attrs["exp_name"] = self.exp_name
            f["meta"].attrs["geometry"] = geom_root
            f.create_group("projections")
            f.create_group("angles")

        return

    def save_2_h5(self, data, angles, idx):
        with h5py.File(os.path.join(self.o_root, f"{self.exp_name}.h5"), "r+") as f:

            f["projections"].create_dataset(f"{str(idx).zfill(5)}", data=data)
            f["angles"].create_dataset(f"{str(idx).zfill(5)}", data=angles)
        return

    def find_corrections(self, correction_parent):
        """Finds correction files in given folder"""
        self.correction_parent = (
            os.path.join(self.root, "Corrections")
            if correction_parent is None
            else correction_parent
        )
        return

    def __call__(self):

        self.init_save_h5()

        # RSD: Could (should) parallise, but is not frequently used code.
        for i, (d_idx, dir) in enumerate(tqdm.tqdm(self.revolution_folders[:-1])):

            assert i == d_idx
            "Revolution folders are not in order!"

            data = np.zeros((self.nproj_360, self.roi[0], self.roi[1]))

            # idx = re.search(" \d-", dir) #RSD : Already done in __init__()
            folder_path = os.path.join(self.root, dir)
            filenames = os.listdir(folder_path)

            angles = self.read_angles(folder_path)

            for j in range(self.nproj_360):

                file = [
                    item
                    for item in filenames
                    if item.endswith(f"{str(j).zfill(5)}.tif")
                ]
                assert len(file) == 1
                "More matches/ Zero matches. Cannot proceed!"

                full_path = os.path.join(folder_path, file[0])

                im = self.load_tif(full_path)
                im = self.normalise_projection(im)
                im = self.remove_defects(im)
                im = self.rotate_projection(im)
                im = self.crop_roi(im, self.roi)
                data[j] = im

            self.save_2_h5(data, angles, i)

        # RSD: Last revolution is special case
        i += 1
        d_idx = self.revolution_folders[-1][0]
        dir = self.revolution_folders[-1][1]
        assert i == d_idx
        "Zero position is not last revolution!"
        folder_path = os.path.join(self.root, dir)
        filenames = os.listdir(folder_path)
        angles = self.read_angles(folder_path)
        data = np.zeros((4, self.roi[0], self.roi[1]))
        for j in range(4):
            file = [
                item for item in filenames if item.endswith(f"{str(j).zfill(5)}.tif")
            ]
            assert len(file) == 1
            "More matches/ Zero matches. Cannot proceed!"
            full_path = os.path.join(folder_path, file[0])
            im = self.load_tif(full_path)
            im = self.normalise_projection(im)
            im = self.remove_defects(im)
            im = self.rotate_projection(im)
            im = self.crop_roi(im, self.roi)
            data[j] = im

        self.save_2_h5(data, angles, i)

        # RSD: Plot anything?
        return
