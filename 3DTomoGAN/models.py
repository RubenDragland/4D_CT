import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os
import sys
import time


class Discriminator3DTomoGAN(nn.Module):
    def __init__(self, hparams=None):
        super().__init__()

        if hparams is None:  # RSD: Not done yet.
            self.hparams = {
                "relu_type": nn.ReLU,
                "input_shape": (32, 64, 64, 64),
                "in_channels": 1,
                "kernel_size": 3,
                "stride": 1,
                "padding": 1,  # Fix Perhaps name stride2_padding
                "groups": 1,  # ? Have to be done in all groups eq 1
                "bias": True,
                "padding_mode": "same",  # Does this work?
                "FC_layers": True,
            }
        else:
            self.hparams = hparams

        if self.hparams["padding_mode"] == "same":
            self.hparams["stride2_padding"] = 1
            # (
            # self.hparams["input_shape"][1] - self.hparams["kernel_size"]
            # ) / 2 + 1  # RSD: Introduce possible bugs. Fix latere
            # RSD: Should work for the first layer, but there is a second layer with S2.
            # RSD: Solution would be to do sequential manually

        self.net = nn.Sequential()
        self.layers = []

        cnn_layers = self.create_CNNS()
        for layer in cnn_layers:
            self.layers.append(layer)
            self.layers.append(self.hparams["relu_type"]())

        fc_layers = self.create_FCS()
        if self.hparams["FC_layers"]:
            # nn.adaptibeavgpool3d Consider RSD
            self.layers.append(nn.AdaptiveAvgPool3d((1, 1, 1)))
            self.layers.append(nn.Flatten())
        for layer in fc_layers:
            self.layers.append(layer)
            self.layers.append(self.hparams["relu_type"]())

        self.net = nn.Sequential(*self.layers)
        return

    def forward(self, x):

        x = self.net(x)
        if not self.hparams[
            "FC_layers"
        ]:  # Current implementation of global max pooling to get a single value regardless of the input size
            x = torch.max(x, dim=2, keepdim=True)
            x = torch.max(x, dim=3, keepdim=True)
            x = torch.max(x, dim=4, keepdim=True)
            # RSD: Check what are spatial dimensions, and what are channels
        return x

    def create_CNNS(self):

        self.C3S1_64 = nn.Conv3d(
            in_channels=self.hparams["in_channels"],
            out_channels=64,
            kernel_size=self.hparams["kernel_size"],
            stride=1,
            padding=self.hparams["padding"],
        )
        self.C3S2_64 = nn.Conv3d(
            in_channels=64,
            out_channels=64,
            kernel_size=self.hparams["kernel_size"],
            stride=2,
            padding=self.hparams["stride2_padding"],
        )
        self.C3S1_128 = nn.Conv3d(
            in_channels=64,
            out_channels=128,
            kernel_size=self.hparams["kernel_size"],
            stride=1,
            padding=self.hparams["padding"],
        )
        self.C3S2_128 = nn.Conv3d(
            in_channels=128,
            out_channels=128,
            kernel_size=self.hparams["kernel_size"],
            stride=2,
            padding=self.hparams["stride2_padding"],
        )
        self.C3S1_256 = nn.Conv3d(
            in_channels=128,
            out_channels=256,
            kernel_size=self.hparams["kernel_size"],
            stride=1,
            padding=self.hparams["padding"],
        )
        self.C3S2_4 = (
            nn.Conv3d(
                in_channels=256,
                out_channels=4,
                kernel_size=self.hparams["kernel_size"],
                stride=2,
                padding=self.hparams[
                    "stride2_padding"
                ],  # RSD: Bug fix. What is the input shape here. Has to be calculated. Possibly give up sequential.
            ),
        )

        return [
            self.C3S1_64,
            self.C3S2_64,
            self.C3S1_128,
            self.C3S2_128,
            self.C3S1_256,
            self.C3S2_4,
        ]  # RSD: Or possibly create an ordered dict.

    def create_FCS(self):
        # RSD: Not fully sure about the below. How to ensure that the output is 1x1x1?
        # self.hparams["relu_type"](),
        if self.hparams["FC_layers"]:
            self.FC_64 = nn.Conv3d(
                in_channels=4,
                out_channels=64,  # RSD; Think this works as FC layer with 64 nodes.
                kernel_size=1,
                stride=1,
                padding=0,
            )
            self.FC_1 = nn.Conv3d(
                in_channels=64,
                out_channels=1,
                kernel_size=1,
                stride=1,
                padding=0,
            )
        else:

            # RSD: Possibly necessary with a softmax or something to get a number.
            self.FC_64 = nn.Linear(
                in_features=1,  # ?,
                out_features=64,
            )
            # self.hparams["relu_type"](),
            self.FC_1 = nn.Linear(
                in_features=64,
                out_features=1,
            )
            # RSD: Possibly necessary with a softmax or something to get a number.
        return [self.FC_64, self.FC_1]


class Generator3DTomoGAN(nn.Module):
    def __init__(self, hparams=None):
        super().__init__()

        if hparams is None:  # RSD: Not done yet.
            self.hparams = {
                "relu_type": nn.ReLU,
                "input_shape": (32, 64, 64, 64),
                "in_channels": 1,
                "kernel_size": 3,
                "stride": 1,
                "padding": 1,  # Fix Perhaps name stride2_padding
                "groups": 1,  # ? Have to be done in all groups eq 1
                "bias": True,
                "padding_mode": "same",  # Does this work?
                "FC_layers": True,
            }
        else:
            self.hparams = hparams

        self.down_sampling1 = []
        self.down_sampling2 = []
        self.down_sampling3 = []
        self.feature_map = []
        self.up_sampling1 = []
        self.up_sampling2 = []
        self.up_sampling3 = []

        self.down_sampling1.append(
            nn.Conv3d(
                in_channels=self.hparams["in_channels"],
                out_channels=8,
                kernel_size=self.hparams["kernel_size"],
                stride=1,
                padding="same",
            )
        )
        self.unet_conv_block(self.down_sampling1, 8, 32)

        self.down_sampling2.append(nn.MaxPool3d(kernel_size=2, stride=2, padding=0))
        self.unet_conv_block(self.down_sampling2, 32, 64)

        self.down_sampling3.append(nn.MaxPool3d(kernel_size=2, stride=2, padding=0))
        self.unet_conv_block(self.down_sampling3, 64, 128)

        self.feature_map.append(nn.MaxPool3d(kernel_size=2, stride=2, padding=0))
        self.feature_map.append(
            nn.Conv3d(
                in_channels=128,
                out_channels=128,
                kernel_size=3,
                stride=1,
                padding="same",
            )
        )
        self.feature_map.append(self.hparams["relu_type"]())
        self.feature_map.append(
            nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False)
        )  # RSD: False?

        self.unet_conv_block(self.up_sampling1, 256, 64)
        self.up_sampling1.append(nn.Upsample(scale_factor=2, mode="trilinear"))

        self.unet_conv_block(self.up_sampling2, 128, 32)
        self.up_sampling2.append(nn.Upsample(scale_factor=2, mode="trilinear"))

        self.unet_conv_block(self.up_sampling3, 64, 32)
        self.up_sampling3.append(
            nn.Conv3d(
                in_channels=32, out_channels=16, kernel_size=1, stride=1, padding="same"
            )
        )
        self.up_sampling3.append(self.hparams["relu_type"]())
        self.up_sampling3.append(
            nn.Conv3d(
                in_channels=16, out_channels=1, kernel_size=1, stride=1, padding="same"
            )
        )

        self.net_down1 = nn.Sequential(*self.down_sampling1)
        self.net_down2 = nn.Sequential(*self.down_sampling2)
        self.net_down3 = nn.Sequential(*self.down_sampling3)
        self.net_feature = nn.Sequential(*self.feature_map)
        self.net_up1 = nn.Sequential(*self.up_sampling1)
        self.net_up2 = nn.Sequential(*self.up_sampling2)
        self.net_up3 = nn.Sequential(*self.up_sampling3)

    def unet_conv_block(self, block, in_ch, out_ch):
        block.append(
            nn.Conv3d(
                in_channels=in_ch,
                out_channels=out_ch,
                kernel_size=3,
                stride=1,
                padding="same",
            )
        )
        block.append(self.hparams["relu_type"]())
        block.append(
            nn.Conv3d(
                in_channels=in_ch,
                out_channels=out_ch,
                kernel_size=3,
                stride=1,
                padding="same",
            )
        )
        block.append(self.hparams["relu_type"]())
        return block

    def forward(self, x):
        skip_connections = []
        x = self.net_down1(x)
        skip_connections.append(x.copy())
        x = self.net_down2(x)
        skip_connections.append(x.copy())
        x = self.net_down3(x)
        skip_connections.append(x.copy())
        x = self.net_feature(x)
        x = torch.cat((x, skip_connections[2]), dim=1)
        x = self.net_up1(x)
        x = torch.cat((x, skip_connections[1]), dim=1)
        x = self.net_up2(x)
        x = torch.cat((x, skip_connections[0]), dim=1)
        x = self.net_up3(x)
        return x


class EncoderDecoder3DTomoGAN(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        pass
