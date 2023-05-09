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
import copy

import tigre.algorithms as algs
import tigre.utilities.gpu as gpu

"""
Here, we will handle tif files obtained at EQNR.
Perhaps make it as a class?
"""

# RSD: Make class to automate this process in case it has changed between scans.


class IndustrialGeometryEQNR:
    def __init__(
        self, default=True, **kwargs
    ):  # RSD: Both nsprg and nspro exist. nsprg for now.
        if default:
            self.values = {
                "DSD": 1350,  # Distance Source Detector (mm)
                "DSO": 930,  # Distance Source Origin (mm)
                "nDetector": np.array([2048, 2048]),  # number of pixels (px)
                "dDetector": np.array([0.2, 0.2]),  # size of each pixel (mm)
                "dVoxel": np.array(
                    [0.124, 0.124, 0.124]
                ),  # Effective voxel pitch (mm) TODO: Check and implement reader.
                "rotation": 0,
            }
        else:
            # try:
            print(kwargs["path"])
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

        golden_geometry.nVoxel = np.array(
            [
                golden_geometry.nDetector[0],
                golden_geometry.nDetector[1],
                golden_geometry.nDetector[1],
            ]
        )  # number of voxels
        golden_geometry.dVoxel = self.values["dVoxel"]
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

        self.geo = golden_geometry
        return

    def __call__(self):
        return self.geo

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
            "ug": f"{root}Ug/ug text",
            "zoom": f"{root}Ug/zoom factor text",
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
        magnification = lambda x: x * self.values["DSO"] / self.values["DSD"]
        self.values["dVoxel"] = np.array(
            [
                magnification(self.values["dDetector"][0]),
                magnification(self.values["dDetector"][1]),
                magnification(self.values["dDetector"][1]),
            ]
        )
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
        rotation=0,
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
            self.geometry, geom_values = geom_obj(), geom_obj.values
        else:
            geom_path = os.path.join(self.root, geometry)
            geom_obj = IndustrialGeometryEQNR(default=False, path=geom_path)
            self.geometry, geom_values = geom_obj.geo, geom_obj.values
            self.geometry.nDetector = np.array([self.roi[0], self.roi[1]])
            self.geometry.nVoxel = np.array([self.roi[0], self.roi[1], self.roi[1]])
            self.geometry.sVoxel = self.geometry.dVoxel * self.geometry.nVoxel
            self.geometry.sDetector = self.geometry.dDetector * self.geometry.nDetector

        self.rotation = geom_values["rotation"] if rotation == 0 else rotation

        try:
            self.angles, names = self.read_angles(
                os.path.join(self.root, f"{exp_name}")
            )  # TODO: Has to be wrong angles
        except:
            print("Except when reading angles!!!")
            self.angles = np.linspace(0, 2 * np.pi, self.number_of_projections)

        self.projections = None

    def __call__(self):
        self.projections = np.zeros(
            (self.number_of_projections, self.roi[0], self.roi[1])
        )

        # self.find_centre_rotation()  # RSD: Not implemented correctly

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

        # proj = (proj - np.min(proj)) / (
        #     np.max(proj) - np.min(proj)
        # )  # RSD: Normalise to avoid negative values in log. However not equal for all projections? Initially performed after I0. Issue, should be corrected.
        # # RSD: If not equal, set neg values to zero.

        proj[proj < 0] = 0
        # RSD: Instead of normalising, set negative values to zero. Cannot be too bad, as it is only a few pixels.

        I0 = np.mean(proj[0:50, 0:50])

        proj = -np.log((proj + tol) / (I0 + tol))

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
        slice_y = slice(shape[0] // 2 - roi[0] // 2, shape[0] // 2 + roi[0] // 2)
        slice_x = slice(
            shape[1] // 2 - roi[1] // 2,
            shape[1] // 2 + roi[1] // 2,
        )
        return proj[slice_y, slice_x]

    def rotate_projection(self, proj):
        """
        Rotating projection according to nsipro angle.
        """
        return nd.rotate(
            proj, self.rotation, reshape=True
        )  # RSD: Try to set this to true

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
        angles = -angles * np.pi / 180
        # RSD: Convert to radians!!! And negative because of the direction of rotation . Tigre and CT disagree.
        names = [item[-9:-4] for item in raw_data.iloc[:, -1]]

        assert angles.dtype == np.float64
        "Wrong conversion. See read_angles(self)"
        assert np.all(angles <= 0)
        "Not all angles are negative. EQNR Cone must be. See read_angles(self)"

        return (
            angles,
            names,
        )

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
            data[idx],
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
        # o_name,
        number_of_projections,
        nrevs=20,
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

        # self.o_name = o_name  # RSD: NB! Distinction between exp_name and o_name. Static is now obsolete ish.

        self.nproj_360 = number_of_projections
        self.nrevs = nrevs
        self.tot_steps = self.nproj_360 * self.nrevs
        self.hor_offset = 0  # RSD: horizontal offset is added to the x slice.

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
        self.revolution_folders = sorted(self.revolution_folders)  # RSD: Sort issue?

        return

    def init_save_h5(self):
        geom_root = os.path.join(self.o_root, f"{self.exp_name}.pkl")
        with open(geom_root, "wb+") as g:
            pkl.dump(self.geometry, g)

        with h5py.File(os.path.join(self.o_root, f"{self.exp_name}.h5"), "w") as f:
            f.create_group("meta")
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

    def assign_data_2_h5(self, data, idx):
        with h5py.File(os.path.join(self.o_root, f"{self.exp_name}.h5"), "r+") as f:
            del f["projections"][f"{str(idx).zfill(5)}"]
            f["projections"].create_dataset(f"{str(idx).zfill(5)}", data=data)
        return

    def change_angles_2_h5(self, angles, idx):
        with h5py.File(os.path.join(self.o_root, f"{self.exp_name}.h5"), "r+") as f:
            del f["angles"][f"{str(idx).zfill(5)}"]
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

    def find_filename_path(self, dir, j, name):
        """Finds the path to the desired projection image"""

        # idx = re.search(" \d-", dir) #RSD : Already done in __init__()
        folder_path = os.path.join(self.root, dir)
        filenames = os.listdir(folder_path)

        file = [item for item in filenames if item.endswith(f"{str(j).zfill(5)}.tif")]
        assert len(file) == 1
        "More matches/ Zero matches. Cannot proceed!"
        assert name == file[0][-9:-4]
        "Wrong angle to file"

        full_path = os.path.join(folder_path, file[0])
        return full_path

    def find_centre_rotation(
        self,
        already_processed=True,
        gs_search=True,
        tol=0.1,
        bounds=(-10, 10),
        ss_size=200,
        depth=100,
    ):
        """
        Aligning rotation of projection image.

        Status:
         Remember radians
         Different results based upon bounds, cropped image size, and plotting (rec) depends on cropped size...
        """

        listGpuNames = gpu.getGpuNames()
        gpuids = gpu.getGpuIds(listGpuNames[0])

        angles = np.zeros(self.nproj_360 * self.nrevs)
        centre_slice = np.zeros((self.nproj_360 * self.nrevs, depth, self.roi[1]))
        print(self.roi)

        if already_processed:
            processed_path = os.path.join(self.o_root, f"{self.exp_name}.h5")
            with h5py.File(processed_path, "r") as f:
                for i in tqdm.trange(self.nrevs):
                    angles[i * self.nproj_360 : (i + 1) * self.nproj_360] = np.array(
                        f["angles"][f"{str(i).zfill(5)}"]
                    )

                    centre_slice[
                        i * self.nproj_360 : (i + 1) * self.nproj_360
                    ] = np.array(
                        f["projections"][f"{str(i).zfill(5)}"][
                            :,
                            self.roi[0] // 2
                            - depth // 2 : self.roi[0] // 2
                            + depth // 2,
                            :,
                        ]
                    )
        else:
            for i in tqdm.trange(self.nrevs):
                folder_path = os.path.join(self.root, self.revolution_folders[i][1])
                (
                    angles[i * self.nproj_360 : (i + 1) * self.nproj_360],
                    names,
                ) = self.read_angles(folder_path)

                for j in range(self.nproj_360):
                    full_path = self.find_filename_path(
                        self.revolution_folders[i][1], j, names[j]
                    )

                    im = self.load_tif(full_path)
                    im0_shape = im.shape
                    im = self.rotate_projection(im)[
                        im0_shape[0] // 2 - depth // 2 : im0_shape[0] // 2 + depth // 2,
                        :,
                    ]

                    centre_slice[i * self.nproj_360 + j] = im

        plt.imshow(centre_slice[centre_slice.shape[0] // 2], cmap="gray")
        plt.show()
        plt.hist(angles, bins=100)
        plt.show()
        print(angles)
        # centre_slice = centre_slice[:, np.newaxis, :]

        def sharpness_score(
            Y, ss_size
        ):  # RSD: Edge in x and y direction. Abs value and sum.
            Y = Y[
                Y.shape[0] // 2 - ss_size // 2 : Y.shape[0] // 2 + ss_size // 2,
                Y.shape[1] // 2 - ss_size // 2 : Y.shape[1] // 2 + ss_size // 2,
            ]
            Y = (Y - np.min(Y)) / (np.max(Y) - np.min(Y))
            Yx = nd.sobel(Y, axis=0)
            Yy = nd.sobel(Y, axis=1)
            return np.sum(np.abs(Yx) ** 2 + np.abs(Yy) ** 2)

        def recursive_gs(offs):
            if offs["x1"][1] > offs["x2"][1]:
                offs["xl"] = offs["x2"]
                offs["x2"] = offs["x1"]

                offs["x1"] = [
                    offs["xl"][0]
                    + (np.sqrt(5) - 1) / 2 * (offs["xu"][0] - offs["xl"][0]),
                    0,
                ]

                slice_geom.COR = offs["x1"][0]
                rec = algs.fdk(
                    centre_slice,
                    slice_geom,
                    angles,
                    gpuids=gpuids,
                )
                offs["x1"][1] = sharpness_score(rec[1], ss_size)

                plt.imshow(
                    rec[rec.shape[0] // 2],
                    cmap="gray",
                )

                plt.title(f"{offs['x1'][0]} SS: {offs['x1'][1]}")
                plt.show()

            else:
                offs["xu"] = offs["x1"]
                offs["x1"] = offs["x2"]

                offs["x2"] = [
                    offs["xu"][0]
                    - (np.sqrt(5) - 1) / 2 * (offs["xu"][0] - offs["xl"][0]),
                    0,
                ]

                slice_geom.COR = offs["x2"][0]
                rec = algs.fdk(
                    centre_slice,
                    slice_geom,
                    angles,
                    gpuids=gpuids,
                )
                offs["x2"][1] = sharpness_score(rec[rec.shape[0] // 2], ss_size)

                plt.imshow(
                    rec[rec.shape[0] // 2],
                    cmap="gray",
                )
                plt.title(f"{offs['x2'][0]} SS: {offs['x2'][1]}")
                plt.show()

            if offs["xu"][0] - offs["xl"][0] < tol:
                self.hor_offset = (offs["xu"][0] + offs["xl"][0]) / 2
                self.geometry.COR = self.hor_offset
                print(f"Sharpnesses: {offs}")
                return self.hor_offset
            else:
                return recursive_gs(offs)

        max_score = 0
        max_offset = 0

        # geo_inst = IndustrialGeometryEQNR()
        # slice_geom = geo_inst()
        slice_geom = copy.deepcopy(self.geometry)
        slice_geom.nDetector = np.array([depth, self.roi[1]])
        slice_geom.sDetector = slice_geom.nDetector * slice_geom.dDetector
        slice_geom.nVoxel = np.array([depth, self.roi[1], self.roi[1]])
        slice_geom.sVoxel = slice_geom.nVoxel * slice_geom.dVoxel

        if gs_search:
            xl, xu = bounds[0], bounds[1]

            d = (np.sqrt(5) - 1) / 2 * (xu - xl)

            x1 = xl + d
            x2 = xu - d

            offs = {"xl": [xl, 0], "x1": [x1, 0], "x2": [x2, 0], "xu": [xu, 0]}

            for o, (k, off) in enumerate(offs.items()):
                slice_geom.COR = off[0]

                rec = algs.fdk(
                    centre_slice,  # input_slice,
                    slice_geom,
                    angles,
                    gpuids=gpuids,
                )
                offs[k][1] = sharpness_score(rec[rec.shape[0] // 2], ss_size)

            print(offs)
            return recursive_gs(offs)

        else:
            # RSD: Consider to use shape of centre_slice instead of roi.

            offset = np.linspace(bounds[0], bounds[1], 11)
            scores = []

            for o, off in enumerate(tqdm.tqdm(offset)):
                slice_geom.COR = off  # np.array([off, 0, 0])

                rec = algs.fdk(
                    centre_slice,  # input_slice,
                    slice_geom,
                    angles,
                    gpuids=gpuids,
                )

                s_score = sharpness_score(rec[rec.shape[0] // 2], ss_size)
                scores.append(s_score)
                if s_score > max_score:
                    max_score = s_score
                    max_offset = off

        print(f"Centre offset: {max_offset}")
        np.save("offsets.npy", offset)
        np.save("scores.npy", np.array(scores))
        self.hor_offset = max_offset
        self.geometry.COR = max_offset

        # RSD: Save to h5 file
        return max_offset

    def process_fast(self):
        def arr_normalise(arr, flat, dark, tol=1e-6):
            """Performs flatfield and darkfield correction on projection image. Also, displays intensity from Beer Lamberts"""
            arr = (arr - dark[np.newaxis]) / (tol + flat[np.newaxis] - dark[np.newaxis])

            arr[arr < 0] = 0
            # RSD: Instead of normalising, set negative values to zero. Cannot be too bad, as it is only a few pixels.

            I0 = np.mean(arr[:, 0:50, 0:50], axis=(1, 2), keepdims=True)

            arr = -np.log((arr + tol) / (I0 + tol))
            return arr

        def arr_remove_defects(arr):
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
                kernel = arr[:, f_index - 1 : f_index + 2, s_index - 1 : s_index + 2]
                arr[:, f_index, s_index] = nd.median_filter(
                    kernel, size=(kernel.shape[0], 3, 3)
                )[:, 1, 1]

            return arr

        def arr_rotate(arr):
            """Rotate stack of images"""
            return nd.rotate(arr, angle=self.rotation, axes=(1, 2), reshape=False)

        self.init_save_h5()
        flat = self.load_tif(os.path.join(self.correction_parent, self.name_flat))
        dark = self.load_tif(os.path.join(self.correction_parent, self.name_dark))

        for i, (d_idx, dir) in enumerate(tqdm.tqdm(self.revolution_folders[:-1])):
            assert i == d_idx
            "Revolution folders are not in order!"

            # data = np.zeros((self.nproj_360, self.roi[0], self.roi[1]))
            folder_path = os.path.join(self.root, dir)
            angles, names = self.read_angles(folder_path)

            full_paths = [
                self.find_filename_path(dir, j, names[j]) for j in range(self.nproj_360)
            ]
            data = np.array(
                [self.load_tif(full_path) for full_path in full_paths]
            )  # RSD: Check if 3d numpy arr with correct shape
            data[:] = arr_normalise(data, flat, dark)
            data[:] = arr_remove_defects(data)
            data[:] = arr_rotate(data)
            data = data[
                :,
                data.shape[1] // 2
                - self.roi[0] // 2 : data.shape[1] // 2
                + self.roi[0] // 2,
                data.shape[2] // 2
                - self.roi[1] // 2 : data.shape[2] // 2
                + self.roi[1] // 2,
            ]

            self.save_2_h5(data, angles, i)

    def __call__(self):
        self.init_save_h5()

        # self.find_centre_rotation() # RSD: TODO: Redo. Make this a separate task so that it may be skipped if it is known.

        # RSD: Could (should) parallise, but is not frequently used code.
        for i, (d_idx, dir) in enumerate(tqdm.tqdm(self.revolution_folders[:-1])):
            assert i == d_idx
            "Revolution folders are not in order!"

            data = np.zeros((self.nproj_360, self.roi[0], self.roi[1]))
            folder_path = os.path.join(self.root, dir)
            angles, names = self.read_angles(folder_path)

            for j in range(self.nproj_360):
                full_path = self.find_filename_path(dir, j, names[j])
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
        angles, names = self.read_angles(folder_path)
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

        return

    def load_processed(self, idx):
        """
        Load processed data from h5 file.
        """
        with h5py.File(os.path.join(self.o_root, f"{self.exp_name}.h5"), "r") as f:
            data = f["projections"][f"{str(idx).zfill(5)}"]
            angles = f["angles"][f"{str(idx).zfill(5)}"]
            return np.array(data), np.array(angles)

    def crop_roi_processed(self, idx: int, roi: tuple) -> np.ndarray:
        """
        Crop ROI for processed data.
        """
        data, angles = self.load_processed(idx)
        dshape = data.shape
        return data[
            :,
            dshape[0] // 2 - roi[0] // 2 : dshape[0] // 2 + roi[0] // 2,
            dshape[-1] // 2 - roi[1] // 2 : dshape[-1] // 2 + roi[1] // 2,
        ]

    def update_roi_geo(self, roi):
        """
        Update ROI in geometry.
        """

        with h5py.File(os.path.join(self.o_root, f"{self.exp_name}.h5"), "r+") as f:
            f["meta"].attrs["roi"] = roi
            geom_root = f["meta"].attrs["geometry"]

        self.roi = roi
        self.geometry["nDetector"] = np.array([roi[0], roi[1]])
        self.geometry["nVoxel"] = np.array([roi[0], roi[1], roi[1]])

        self.geometry["sDetector"] = (
            self.geometry["nDetector"] * self.geometry["dDetector"]
        )
        self.geometry["sVoxel"] = self.geometry["nVoxel"] * self.geometry["dVoxel"]

        with open(geom_root, "w") as g:
            pkl.dump(self.geometry, g)

        return

    def crop_roi_processed_projections(self, roi):
        """
        Crop ROI for processed data.
        """

        self.roi = roi

        self.update_roi_geo(roi)

        for idx in tqdm.trange(self.nrevs):
            data, angles = self.load_processed(idx)

            data = self.crop_roi_processed(idx, roi)

            self.assign_data_2_h5(data, idx)

        return
