import torch
import torch.nn as nn
import numpy as np

import imageio


binary_cross_entropy = nn.BCEWithLogitsLoss()  # TomoGAN uses BinaryCrossEntropy
mean_squared_error = nn.MSELoss()


# RSD: Use some time understanding this function
# Real images should have a classification prob of 1,
# while fake images should have a classification prob of 0
# Pytorch uses (output, target)
def discriminator_loss(real_output, fake_output):
    real_loss = binary_cross_entropy(real_output, torch.ones_like(real_output))
    fake_loss = binary_cross_entropy(fake_output, torch.zeros_like(fake_loss))
    total_loss = real_loss + fake_loss
    return total_loss


# RSD: Believe this is binary cross entropy for generator
# RSD: In this case, one would like to maximize fake_output probability
def adversarial_loss(fake_output):
    return binary_cross_entropy(fake_output, torch.ones_like(fake_output))


# RSD: This part is unoptimised. Might be slow, and is a part of a loop...
def feature_extraction_iteration_loss(feature_extractor, X_vgg, Y_vgg, i):

    # RSD: This expansion may be unnecessary

    X_vgg_extracted_x = feature_extractor(
        torch.cat(
            [
                X_vgg[:, :, i, :, :],
                X_vgg[:, :, i, :, :],
                X_vgg[:, :, i, :, :],
            ],
            dim=1,
        )
    )  # N, C, D, H, W

    X_vgg_extracted_y = feature_extractor(
        torch.cat(
            [
                X_vgg[:, :, :, i, :],
                X_vgg[:, :, :, i, :],
                X_vgg[:, :, :, i, :],
            ],
            dim=1,
        )
    )  # N, C, D, H, W

    X_vgg_extracted_z = feature_extractor(
        torch.cat(
            [
                X_vgg[:, :, :, :, i],
                X_vgg[:, :, :, :, i],
                X_vgg[:, :, :, :, i],
            ],
            dim=1,
        )
    )  # N, C, D, H, W

    Y_vgg_extracted_x = feature_extractor(
        torch.cat(
            [Y_vgg[:, :, i, :, :], Y_vgg[:, :, i, :, :], Y_vgg[:, :, i, :, :]],
            dim=1,
        )
    )  # N, C, D, H, W

    Y_vgg_extracted_y = feature_extractor(
        torch.cat(
            [Y_vgg[:, :, :, i, :], Y_vgg[:, :, :, i, :], Y_vgg[:, :, :, i, :]],
            dim=1,
        )
    )  # N, C, D, H, W

    Y_vgg_extracted_z = feature_extractor(
        torch.cat(
            [Y_vgg[:, :, :, :, i], Y_vgg[:, :, :, :, i], Y_vgg[:, :, :, :, i]],
            dim=1,
        )
    )  # N, C, D, H, W

    loss_x = mean_squared_error(
        X_vgg_extracted_x.reshape(-1), Y_vgg_extracted_x.reshape(-1)
    )
    loss_y = mean_squared_error(
        X_vgg_extracted_y.reshape(-1), Y_vgg_extracted_y.reshape(-1)
    )
    loss_z = mean_squared_error(
        X_vgg_extracted_z.reshape(-1), Y_vgg_extracted_z.reshape(-1)
    )

    return loss_x + loss_y + loss_z


def save2img(d_img, fn):
    _min, _max = d_img.min(), d_img.max()
    if np.abs(_max - _min) < 1e-4:
        img = np.zeros(d_img.shape)
    else:
        img = (d_img - _min) * 255.0 / (_max - _min)

    img = img.astype("uint8")
    imageio.imwrite(fn, img)
