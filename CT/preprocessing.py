import tigre
import numpy as np
import scipy.ndimage as nd
from PIL import Image
import os
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import h5py

"""
Here, we will handle tif files obtained at EQNR.
Perhaps make it as a class?
"""


class IndustrialGeometryEQNR:
    def __init__():

        golden_geometry = tigre.geometry(mode="cone", default=True)

        golden_geometry.DSD = 1350  # Distance Source Detector (mm)
        golden_geometry.DSO = 930  # Distance Source Origin (mm)

        golden_geometry.nDetector = np.array([2048, 2048])  # number of pixels (px)
        golden_geometry.dDetector = np.array([0.2, 0.2])  # size of each pixel (mm)
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

        golden_geometry.rotDetector = np.array(
            [270 / 180 * np.pi, 0, 0]
        )  # Rotation of detector

        return golden_geometry


class ProjectionsEQNR:
    def __init__(
        self,
        root,  # RSD: Consider to create correct folder here.
        exp_name,
        number_of_projections,
        correction_parent=None,
        name_flat="gain0.tif",
        name_dark="offset.tif",
        dtype=np.uint8,  # Not implemented
        geometry=None,
        roi=None,
        metallic_mean_n=0,
        rotation=0,
    ):
        self.root = root
        self.exp_name = exp_name
        self.number_of_projections = number_of_projections
        self.p_roots = [
            os.path.join(root, f"{exp_name}{str(i).zfill(5)}.tif")
            for i in range(number_of_projections)
        ]

        self.correction_parent = (
            Path(root).parent[0] if correction_parent == None else correction_parent
        )
        self.name_flat = name_flat
        self.name_dark = name_dark

        if roi is None:
            self.roi = self.load_tif(
                os.path.join(self.correction_parent, name_flat)
            ).shape
            print(self.roi)
        else:
            self.roi = roi

        if geometry is None:
            self.geometry = tigre.geometry(
                mode="cone",
                default=True,
                nVoxel=(self.roi[0], self.roi[0], self.roi[1]),
            )
        else:
            self.geometry = geometry

        self.rotation = rotation

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

        self.dtype = dtype
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

        # self.save_h5()
        self.plot_projection(0)

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

    def save_h5(self):
        """Saving projections as h5 file with geometry data for reconstruction"""
        with h5py.File(os.path.join(self.root, f"{self.exp_name}.h5"), "w") as f:
            f.create_dataset("projections", data=self.projections)
            f.create_dataset("angles", data=self.angles)
            f.create_dataset("geometry", data=self.geometry)
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

        return

    def plot_projection(self, index=0):
        """Plot projection at index"""
        plt.imshow(self.projections[index], cmap="gray")
        plt.colorbar()
        plt.show()
        return
