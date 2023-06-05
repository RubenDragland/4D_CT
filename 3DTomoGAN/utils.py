import torch
import torch.nn as nn
import numpy as np
import time
import scipy as sp
import tqdm as tqdm
import torch.nn.functional as F

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


def gradient_penalty_discriminator_loss(real_output, fake_output, X, Y, discriminator):
    """
    Believe this is the correct implementation of wasserstein distance. Test for discussion.
    """
    epsilon = np.random.uniform(0, 1, size=(X.shape[0], 1, 1, 1, 1))
    I = epsilon * X + (1 - epsilon) * Y
    lambda_d = 10
    I.to(real_output.device)
    I.requires_grad_(True)
    wasserstein = np.mean(fake_output - real_output)
    output = discriminator(I)
    gradients = torch.autograd.grad(
        outputs=output,
        inputs=I,
        grad_outputs=torch.ones(I.size()).to(real_output.device),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]

    gradient_penalty = lambda_d * torch.mean((gradients.norm(2, dim=1) - 1) ** 2)

    return wasserstein + gradient_penalty


def wasserstein_distance_adv(Dx):
    return -torch.mean(Dx)


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


def slice_feature_extraction_ssim_loss(feature_extractor, X_perc, Y_perc):
    X_perc = torch.unsqueeze(X_perc, 0) if len(X_perc.shape) == 3 else X_perc
    Y_perc = torch.unsqueeze(Y_perc, 0) if len(Y_perc.shape) == 3 else Y_perc

    X_perc = feature_extractor(X_perc)  # N, C, H, W
    Y_perc = feature_extractor(Y_perc)  # N, C, H, W

    return 1 - torch_ssim(Y_perc, X_perc)


def perc_indexer_x(Z, i):
    return Z[:, :, i, :, :]


def perc_indexer_y(Z, i):
    return Z[:, :, :, i, :]


def perc_indexer_z(Z, i):
    return Z[:, :, :, :, i]


def perc_slice_loop(feature_extractor, indexer, preprocess, X, Y, slices):
    loss = 0
    for i in range(slices):  # RSD: Drop loop, instead merge slices into batch dimension
        X_perc = indexer(X, i)
        Y_perc = indexer(Y, i)

        Y_perc = torch.squeeze(
            Y_perc.expand(Y_perc.shape[0], 3, Y_perc.shape[2], Y_perc.shape[3])
        )
        Y_perc = preprocess(Y_perc)

        X_perc = torch.squeeze(
            X_perc.expand(X_perc.shape[0], 3, X_perc.shape[2], X_perc.shape[3])
        )
        X_perc = preprocess(X_perc)

        loss += slice_feature_extraction_loss(feature_extractor, X_perc, Y_perc)
        # loss += slice_feature_extraction_ssim_loss(feature_extractor, X_perc, Y_perc)
        # RSD: Now evaluating perceptual loss using SSIM instead of mse. Does it make sense? Or is it smør på flesk
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


def torch_ssim(I, J, c1=0.01**2, c2=0.03**2):
    mean_I = torch.mean(I)
    mean_J = torch.mean(J)

    var_I = torch.var(I)
    var_J = torch.var(J)

    # cov_IJ = torch.mean((I - mean_I) * (J - mean_J))

    ssim = (
        (2 * mean_I * mean_J + c1)
        * (2 * torch.mean((I - mean_I) * (J - mean_J)) + c2)
        / ((mean_I**2 + mean_J**2 + c1) * (var_I + var_J + c2))
    )
    return ssim


def torch_psnr(I, J, norm=False):
    normalise = lambda img: (img - img.min()) / (img.max() - img.min())
    if norm:
        I = normalise(I)
        J = normalise(J)
    mse = torch.mean((I - J) ** 2)
    max_val = torch.max(I)
    psnr = 10 * torch.log10(max_val**2 / mse)
    return psnr


def calc_ssim(I, J, c1=0.01**2, c2=0.03**2, norm=False):
    normalise = lambda img: (img - img.min()) / (img.max() - img.min())
    if norm:
        I = normalise(I)
        J = normalise(J)

    mean_I = np.mean(I)
    mean_J = np.mean(J)

    var_I = np.var(I)
    var_J = np.var(J)

    cov_IJ = np.mean((I - mean_I) * (J - mean_J))

    ssim = (
        (2 * mean_I * mean_J + c1)
        * (2 * cov_IJ + c2)
        / ((mean_I**2 + mean_J**2 + c1) * (var_I + var_J + c2))
    )
    return ssim


def calc_psnr(I, J):
    mse = np.mean((I - J) ** 2)
    max_val = np.max(I)
    psnr = 10 * np.log10(max_val**2 / mse)
    return psnr


def calc_mssim(I, J, c1=0.01**2, c2=0.03**2, k=11):
    """
    Mean-SSIM, Retrieved from SSIM-PyTorch on Github
    """

    def gaussian(window_size, sigma):
        import math

        """
        Generates a list of Tensor values drawn from a gaussian distribution with standard
        diviation = sigma and sum of all elements = 1.

        Length of list = window_size
        """
        gauss = torch.Tensor(
            [
                math.exp(-((x - window_size // 2) ** 2) / float(2 * sigma**2))
                for x in range(window_size)
            ]
        )
        return gauss / gauss.sum()

    def create_window(window_size, channel=1):
        # Generate an 1D tensor containing values sampled from a gaussian distribution
        _1d_window = gaussian(window_size=window_size, sigma=1.5).unsqueeze(1)

        # Converting to 2D
        _2d_window = _1d_window.mm(_1d_window.t()).float().unsqueeze(0).unsqueeze(0)

        window = torch.Tensor(
            _2d_window.expand(channel, 1, window_size, window_size).contiguous()
        )

        return window

    def ssim(
        img1,
        img2,
        val_range,
        window_size=11,
        window=None,
        size_average=True,
        full=False,
    ):
        L = val_range  # L is the dynamic range of the pixel values (255 for 8-bit grayscale images),
        pad = window_size // 2

        normalise = lambda img: (img - img.min()) / (img.max() - img.min())

        img1 = normalise(img1)
        img2 = normalise(img2)
        L1 = torch.max(img1)
        L2 = torch.min(img1)
        L = L1 - L2

        try:
            _, channels, height, width = img1.size()
        except:
            channels, height, width = img1.size()

        # if window is not provided, init one
        if window is None:
            real_size = min(
                window_size, height, width
            )  # window should be atleast 11x11
            window = create_window(real_size, channel=channels).to(img1.device)

        # calculating the mu parameter (locally) for both images using a gaussian filter
        # calculates the luminosity params
        mu1 = F.conv2d(img1, window, padding=pad, groups=channels)
        mu2 = F.conv2d(img2, window, padding=pad, groups=channels)

        mu1_sq = mu1**2
        mu2_sq = mu2**2
        mu12 = mu1 * mu2

        # now we calculate the sigma square parameter
        # Sigma deals with the contrast component
        sigma1_sq = F.conv2d(img1 * img1, window, padding=pad, groups=channels) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, window, padding=pad, groups=channels) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, window, padding=pad, groups=channels) - mu12

        # Some constants for stability
        C1 = 0.01**2 * L**2  # NOTE: Removed L from here (ref PT implementation)
        C2 = 0.03**2 * L**2

        contrast_metric = (2.0 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2)
        contrast_metric = torch.mean(contrast_metric)

        numerator1 = 2 * mu12 + C1
        numerator2 = 2 * sigma12 + C2
        denominator1 = mu1_sq + mu2_sq + C1
        denominator2 = sigma1_sq + sigma2_sq + C2

        ssim_score = (numerator1 * numerator2) / (denominator1 * denominator2)

        if size_average:
            ret = ssim_score.mean()
        else:
            ret = ssim_score.mean(1).mean(1).mean(1)

        if full:
            return ret, contrast_metric

        return ret

    window = create_window(k, 1)

    tensortransform = lambda x: torch.from_numpy(x).unsqueeze(0)

    I = tensortransform(I)
    J = tensortransform(J)

    mssim, contrasts = ssim(
        I, J, 1.0, window=window, window_size=k, size_average=True, full=True
    )

    mssim = mssim.item()
    contrasts = contrasts.item()

    return mssim, contrasts


def FSC(gt, elem, sizes=(256, 256, 256)):
    gt_k = sp.fft.fftshift(sp.fft.fftn(gt)).flatten()
    elem_k = sp.fft.fftshift(sp.fft.fftn(elem)).flatten()

    X, Y, Z = np.meshgrid(np.arange(sizes[0]), np.arange(sizes[1]), np.arange(sizes[2]))

    radius = np.sqrt(
        (X - sizes[0] // 2) ** 2 + (Y - sizes[1] // 2) ** 2 + (Z - sizes[2] // 2) ** 2
    ).flatten()

    uniques = np.unique(radius)
    print(uniques.shape)

    gt_dict = {}
    elem_dict = {}
    for u in uniques:
        gt_dict[u] = []
        elem_dict[u] = []

    for i, u in enumerate(tqdm.tqdm(radius)):
        gt_dict[u].append(gt_k[i])
        elem_dict[u].append(elem_k[i])

    uniques = np.sort(uniques)

    FSCR = np.zeros_like(uniques, dtype=np.complex64)

    for i, u in enumerate(tqdm.tqdm(uniques)):
        # gt_kr = gt_k[np.where(radius == u)]
        # elem_kr = elem_k[np.where(radius == u)]
        gt_kr = np.array(gt_dict[u])
        elem_kr = np.array(elem_dict[u])

        upper = np.sum(gt_kr * np.conj(elem_kr))

        lower = np.sqrt(np.sum(np.abs(gt_kr) ** 2) * np.sum(np.abs(elem_kr) ** 2))

        FSCR[i] = upper / lower

    return FSCR, uniques


def evaluate_recs(x, y, normalise=False):
    # Normalises between 0 and 1
    if normalise:
        x = (x - np.min(x)) / (np.max(x) - np.min(x))
        y = (y - np.min(y)) / (np.max(y) - np.min(y))

    ssim = calc_ssim(x, y)
    psnr = calc_psnr(x, y)
    return ssim, psnr


def calc_sobel(crossection):
    import scipy.ndimage as nd

    crossection = np.squeeze(crossection)
    grads = []
    for i in range(len(crossection.shape)):
        grads.append(nd.sobel(crossection, axis=i))
    # grad_x = nd.sobel(crossection, axis=0)
    # grad_y = nd.sobel(crossection, axis=1)
    # grad_z = nd.sobel(crossection, axis=2)

    grads = np.array(grads)
    return np.sqrt(np.sum(grads**2, axis=0))


def TV_score(x):
    grad = calc_sobel(x)
    return np.sum(grad)
