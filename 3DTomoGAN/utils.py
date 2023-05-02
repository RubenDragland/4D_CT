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
def discriminator_loss(
    real_output, fake_output, classification_loss=nn.BCEWithLogitsLoss()
):
    """Classifies real and fake images and computes the loss for the discriminator"""
    real_loss = classification_loss(real_output, torch.ones_like(real_output))
    fake_loss = classification_loss(fake_output, torch.zeros_like(fake_output))
    total_loss = real_loss + fake_loss
    return total_loss


# RSD: Believe this is binary cross entropy for generator
# RSD: In this case, one would like to maximize fake_output probability
def adversarial_loss(fake_output, classification_loss=nn.BCEWithLogitsLoss()):
    """Computes the adversarial loss for the generator"""
    return classification_loss(fake_output, torch.ones_like(fake_output))


# RSD: This part is unoptimised. Might be slow, and is a part of a loop...


def slice_feature_extraction_loss(feature_extractor, X_perc, Y_perc):
    X_perc = torch.unsqueeze(X_perc, 0) if len(X_perc.shape) == 3 else X_perc
    Y_perc = torch.unsqueeze(Y_perc, 0) if len(Y_perc.shape) == 3 else Y_perc

    X_perc = feature_extractor(X_perc)  # N, C, H, W
    Y_perc = feature_extractor(Y_perc)  # N, C, H, W

    return mean_squared_error(X_perc.reshape(-1), Y_perc.reshape(-1))


def perc_indexer_x(Z, i):
    return Z[:, :, i, :, :]


def perc_indexer_y(Z, i):
    return Z[:, :, :, i, :]


def perc_indexer_z(Z, i):
    return Z[:, :, :, :, i]


def perc_slice_loop(feature_extractor, indexer, preprocess, X, Y, slices):
    loss = 0
    for i in range(slices):
        X_perc = indexer(X, i)
        Y_perc = indexer(Y, i)

        # Y_perc = torch.squeeze(
        #     torch.stack(
        #         [Y_perc, Y_perc, Y_perc],
        #         dim=1,
        #     )
        # )
        Y_perc = torch.squeeze(
            Y_perc.expand(Y_perc.shape[0], 3, Y_perc.shape[2], Y_perc.shape[3])
        )
        Y_perc = preprocess(Y_perc)

        # X_perc = torch.squeeze(
        #     torch.stack(
        #         [X_perc, X_perc, X_perc],
        #         dim=1,
        #     )
        # )
        X_perc = torch.squeeze(
            X_perc.expand(X_perc.shape[0], 3, X_perc.shape[2], X_perc.shape[3])
        )
        X_perc = preprocess(X_perc)

        loss += slice_feature_extraction_loss(feature_extractor, X_perc, Y_perc)
    return loss


def save2img(d_img, fn):
    _min, _max = d_img.min(), d_img.max()
    if np.abs(_max - _min) < 1e-4:
        img = np.zeros(d_img.shape)
    else:
        img = (d_img - _min) * 255.0 / (_max - _min)
        # img = d_img * 255.0

    img = img.astype("uint8")
    imageio.imwrite(fn, img)


def calc_ssim(I, J, c1=0.01**2, c2=0.03**2):
    mean_I = np.mean(I)
    mean_J = np.mean(J)

    var_I = np.var(I)
    var_J = np.var(J)

    cov_IJ = np.mean(I * J) - mean_I * mean_J
    print(cov_IJ)
    cov_IJ = np.mean((I - mean_I) * (J - mean_J))
    print(cov_IJ)

    ssim = (
        (2 * mean_I * mean_J + c1)
        * (2 * cov_IJ + c2)
        / ((mean_I**2 + mean_J**2 + c1) * (var_I**2 + var_J**2 + c2))
    )
    return ssim


def calc_psnr(I, J):
    mse = np.mean((I - J) ** 2)
    max_val = np.max(I)
    psnr = 10 * np.log10(max_val**2 / mse)
    return psnr
