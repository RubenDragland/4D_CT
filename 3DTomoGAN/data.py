import os
import sys
import time
import numpy as np
import torch

# from torchvision.io import read_image
import imageio
from torch.utils.data import Dataset
import torchio as tio

# import torchvision.transforms.functional as TF
import torchvision.transforms as transforms
import h5py


def inverse_transform(tensor):
    return 1 - tensor.copy()


def random_inverse_transform(tensor, p=0.5):
    if np.random.random() > p:
        return inverse_transform(tensor)
    else:
        return tensor


def tio_identity(tensor):
    return torch.clone(tensor)


includes = ("data", "target")
inverse_transform = tio.Lambda(inverse_transform, include=includes)
random_inverse_transform = tio.Lambda(random_inverse_transform, include=includes)

# 31% change of not flipping
# Prob could be changed as a param.
# RSD: Issue: input and target must be flipped the same way
advanced_flipping = tio.Compose(
    [
        tio.RandomFlip(0, 0.31, include=includes),
        tio.RandomFlip(1, 0.31, include=includes),
        tio.RandomFlip(2, 0.31, include=includes),
    ]
)

flipping_transforms = tio.OneOf(
    {
        tio.Lambda(tio_identity, include=includes): 0.20,
        tio.RandomFlip(0, 1, include=includes): 0.20,
        tio.RandomFlip(1, 1, include=includes): 0.20,
        tio.RandomFlip(2, 1, include=includes): 0.20,
        advanced_flipping: 0.20,
    }
)


# RSD: Figure out the different transforms.
basic_transforms = tio.Compose(
    [
        tio.RescaleIntensity((0, 1), include=includes),
        random_inverse_transform,  # RSD: Not when transfer learning
        flipping_transforms,
        tio.RandomAffine(
            scales=(1, 1),
            degrees=(0, 360, 0, 360, 0, 360),
            isotropic=True,
            include=includes,
        ),
    ],
)
# RSD: Consider how rotation should be conducted

advanced_transforms = tio.Compose(
    [
        basic_transforms,
        random_inverse_transform,
        tio.RandomElasticDeformation(num_control_points=(10, 10, 10), locked_borders=1),
        # tio.RandomMotion(
        #     num_transforms=1, max_displacement=(5, 5, 5), locked_borders=1
        # ),
        tio.OneOf(
            {
                tio.RandomNoise(std=(0, 0.1)): 0.5,
                tio.RandomBlur(std=(0, 1)): 0.5,
            }
        ),
    ]
)


class Dataset3D(Dataset):
    transforms_dict = {
        "basic": basic_transforms,
        "advanced": advanced_transforms,
    }

    def __init__(self, filename, img_dir_root, hparams):
        self.root = os.path.join(img_dir_root, f"{filename}.h5")
        self.filename = filename
        self.hparams = hparams
        self.transform = Dataset3D.transforms_dict[hparams["transforms"]]  # [transform]
        self.target_transform = Dataset3D.transforms_dict[
            hparams["transforms"]
        ]  # [target_transform]

    def __getitem__(self, index):
        dimensions = h5py.File(self.root, "r")["noisy3D"][str(index).zfill(5)].shape
        size = self.hparams["psz"]

        crop_centres = self.random_crop_h5py(dimensions, size)
        # RSD: Hard-code size. Else include hparam in Dataset3D.

        data = self.load_cropped("noisy3D", index, crop_centres, size)
        target = self.load_cropped("target3D", index, crop_centres, size)

        # data = torch.from_numpy(
        #     np.array(
        #         h5py.File(self.root, "r")["noisy3D"][str(index).zfill(5)][
        #             224:-224, 96:-96, 96:-96
        #         ]
        #     )
        # )
        # # RSD: Temp slicing.
        # target = torch.from_numpy(
        #     np.array(
        #         h5py.File(self.root, "r")["target3D"][str(index).zfill(5)][
        #             224:-224, 96:-96, 96:-96
        #         ]
        #     )
        # )
        # if self.transform is not None:
        #     data = np.squeeze(self.transform(data[np.newaxis, :]))
        # if self.target_transform is not None:
        #     target = np.squeeze(self.target_transform(target[np.newaxis, :]))

        # RSD: For now the same transforms have to be applied to data and target.
        combined_dict = {"data": data[np.newaxis, :], "target": target[np.newaxis, :]}
        combined_dict = self.transform(
            combined_dict,
        )  # include=("data", "target"))

        return np.squeeze(combined_dict["data"]), np.squeeze(combined_dict["target"])

    def __len__(self):
        return len(
            h5py.File(self.root, "r")["target3D"].keys()
        )  # RSD: Hope this will remain valid and that it works.

    def load_cropped(self, key, index, crop_centres, size):
        return torch.from_numpy(
            np.array(
                h5py.File(self.root, "r")[key][str(index).zfill(5)][
                    crop_centres[0] - size // 2 : crop_centres[0] + size // 2,
                    crop_centres[1] - size // 2 : crop_centres[1] + size // 2,
                    crop_centres[2] - size // 2 : crop_centres[2] + size // 2,
                ]
            )
        )

    # More variation with uniform sampling.
    def random_crop_h5py(self, dimensions: tuple, size: int, gaussian=True) -> tuple:
        def find_bounds(dimensions, size):
            low = -dimensions // 2 + size // 2 + 1
            high = dimensions // 2 - size // 2 - 1
            return low, high

        def find_std(dimensions, size):
            low, high = find_bounds(dimensions, size)
            return (high - low) * 0.05

        def normal_crop(rng, dimensions, size):
            return (
                rng.normal(0, find_std(dimensions, size), 1)
                .astype(int)
                .clip(*find_bounds(dimensions, size))
            )[0]

        rng = np.random.default_rng()

        if gaussian:
            crop_x = normal_crop(rng, dimensions[0], size)
            crop_y = normal_crop(rng, dimensions[1], size)
            crop_z = normal_crop(rng, dimensions[2], size)
        else:
            crop_x = int(rng.uniform(*find_bounds(dimensions[0], size)))
            crop_y = int(rng.uniform(*find_bounds(dimensions[1], size)))
            crop_z = int(rng.uniform(*find_bounds(dimensions[2], size)))

        crop_x = dimensions[0] // 2 + crop_x
        crop_y = dimensions[1] // 2 + crop_y
        crop_z = dimensions[2] // 2 + crop_z
        # RSD: Hope within bounds
        # RSD:print when training to look at distribution. Too narrow? Go for uniform now?

        return crop_x, crop_y, crop_z


# class Dataloader3D(DataLoader):
#     # not necessarily need for an own class
#     pass


# Split data sets

"""
To make custom transform.
def custom_transform():
    return


custom_trans = tio.Lambda(custom_transform)

"""

"""
Ideas to transformations:

Definetly:
ToTensor --> In the loading of the data. 
Normalise (0,1)
Crop --> In the loading of the data. 
Flip 
Rotate --> Must consider to rotate the first axis or all. The latter would move undersampling artefacts. But believe this was done in the successful trial
Inverse
Random Elastic Deformation (Non-linear, weakly)

Perhaps:
Random Motion
Random Noise If FDK
Random Blur? If Iterative

Questions: How to compose?
Only compose
One of some?

"""

# Uses one of, could also perform more flips
