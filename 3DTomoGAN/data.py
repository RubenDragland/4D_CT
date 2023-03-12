import os
import sys
import time
import numpy as np
import torch

# from torchvision.io import read_image
import imageio
from torch.utils.data import Dataset

# import torchvision.transforms.functional as TF
import torchvision.transforms as transforms
import h5py

# RSD: Ish initialised.


# RSD: Need plan for cropping etc. Do this as target transform and transform? Possibly.

# RSD: From h5 to dataset is the plan.


class Dataset3D(Dataset):
    def __init__(
        self,
        filename,
        img_dir_root,
        transform=None,  # transforms.Compose([transforms.ToTensor()]),
        target_transform=None,  # transforms.Compose([transforms.ToTensor()]),
    ):
        self.root = os.path.join(img_dir_root, filename)  # Change so h5 is implied
        self.filename = filename
        self.transform = transform
        self.target_transform = target_transform
        # self.data = []
        # self.targets = []
        # self._load_data()

    # Not necessary for h5 files
    # def _load_data(self):
    #     for file in os.listdir(self.root):
    #         if file.endswith(".tif"):
    #             self.data.append(os.path.join(self.root, file))
    #             self.targets.append(os.path.join(self.root, file))

    def __getitem__(self, index):
        data = torch.from_numpy(
            np.array(
                h5py.File(self.root, "r")["noisy3D"][str(index).zfill(5)][
                    240:-240, 112:-112, 112:-112
                ]
            )
        )
        # RSD: Temp slicing.
        target = torch.from_numpy(
            np.array(
                h5py.File(self.root, "r")["target3D"][str(index).zfill(5)][
                    240:-240, 112:-112, 112:-112
                ]
            )
        )
        if self.transform is not None:
            data = self.transform(data)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return data, target

    def __len__(self):
        return len(
            h5py.File(self.root, "r")["target3D"].keys()
        )  # RSD: Hope this will remain valid and that it works.


# class Dataloader3D(DataLoader):
#     # not necessarily need for an own class
#     pass


# Split data sets
