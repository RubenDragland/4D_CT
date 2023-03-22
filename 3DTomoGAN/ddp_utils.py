import torch
from torch import nn as nn
from torch.utils.data import DataLoader
import numpy as np
import logging
import utils
import os
import sys
import time


def init_data(d_set, b_size, num_gpus):
    """Distributes data across gpus."""
    pass
    sampler = torch.utils.data.distributed.DistributedSampler(d_set)
    d_loader = DataLoader(
        dataset=d_set,
        batch_size=b_size,
        shuffle=False,
        num_workers=4 * num_gpus,
        pin_memory=True,  # RSD? Believe this is performance > memory
        sampler=sampler,
    )
    return d_loader


def init_model(model, rank):
    """Distributes model across gpus."""
    model = nn.parallel.DistributedDataParallel(model, device_ids=[rank])
    torch.cuda.set_device(rank)
    model.cuda(rank)
    return model


def generator_loop(
    train_dataloader,
    gen_optim,
    generator,
    discriminator,
    feature_extractor,
    feature_preprocess,
    criterion_mse,
    criterion_bce,
    rank,
    args,
    epoch,
    time_begin,
):
    """Trains the generator."""
    for _ge in range(args.itg):

        X, Y = next(iter(train_dataloader))  # RSD: Check if correct unpacking.
        X, Y = torch.unsqueeze(X, dim=1).cuda(rank, non_blocking=True), torch.unsqueeze(
            Y, dim=1
        ).cuda(rank, non_blocking=True)

        # Train Generator
        gen_optim.zero_grad()
        X = generator(X)  # Generator works
        # Calculate loss
        loss_mse = criterion_mse(X, Y)  # RSD: Changed because loss in process.

        logging.info("Loss MSE: %f" % loss_mse)

        loss_adv = utils.adversarial_loss(discriminator(X), criterion_bce)
        logging.info("Loss adv: %f" % loss_adv)

        mem = torch.cuda.mem_get_info()
        logging.debug("Before perc: " + str((mem[1] - mem[0]) / 1024 / 1024 / 1024))

        perc_loss = 0

        perc_loss += utils.perceptual_loss(
            feature_extractor,
            utils.perc_indexer_x,
            feature_preprocess,
            X,
            Y,
            X.shape[2],
        )
        perc_loss += utils.perceptual_loss(
            feature_extractor,
            utils.perc_indexer_y,
            feature_preprocess,
            X,
            Y,
            X.shape[3],
        )
        perc_loss += utils.perceptual_loss(
            feature_extractor,
            utils.perc_indexer_z,
            feature_preprocess,
            X,
            Y,
            X.shape[4],
        )

        mem = torch.cuda.mem_get_info()

    logging.info("Perc loss: %f" % perc_loss)
    logging.debug("After Gen: " + str((mem[1] - mem[0]) / 1024 / 1024 / 1024))

    generator_loss = (
        args.lmse * loss_mse + args.ladv * loss_adv + args.lperc * perc_loss
    )
    logging.debug("Generator loss: %f" % generator_loss)

    generator_loss.backward()
    gen_optim.step()

    itr_prints_gen = (
        "[Info] Epoch: %05d, %.2fs/it, gloss: %.2f (mse%.3f, adv%.3f, perc:%.3f)"
        % (
            epoch,
            (time.time() - time_begin) / args.itg,
            generator_loss.cpu().detach().numpy(),
            loss_mse.cpu().detach().numpy().mean() * args.lmse,
            loss_adv.cpu().detach().numpy().mean() * args.ladv,
            perc_loss.cpu().detach().numpy().mean() * args.lperc,
        )
    )

    return itr_prints_gen  # RSD: Check whether something has to be returned. In that case, return gen loss and do backprop outside.


def discriminator_loop(
    train_dataloader,
    disc_optim,
    generator,
    discriminator,
    criterion_bce,
    rank,
    args,
    epoch,
    time_begin_disc,
    time_begin_gen,
    itr_prints_gen,
):
    """Trains the discriminator."""

    for _de in range(args.itd):

        X, Y = next(iter(train_dataloader))

        X, Y = torch.unsqueeze(X, dim=1).cuda(rank, non_blocking=True), torch.unsqueeze(
            Y, dim=1
        ).cuda(rank, non_blocking=True)

        disc_optim.zero_grad()
        X = generator(X)

        discriminator_real = discriminator(Y)
        discriminator_fake = discriminator(X)
        discriminator_loss = utils.discriminator_loss(
            discriminator_real, discriminator_fake, criterion_bce
        )
        logging.info("Discriminator loss: %f" % discriminator_loss)

        discriminator_loss.backward()
        disc_optim.step()

    disc_optim.zero_grad()

    print(
        "%s; dloss: %.2f (r%.3f, f%.3f), disc_elapse: %.2fs/itr, gan_elapse: %.2fs/itr"
        % (
            itr_prints_gen,
            discriminator_loss.cpu().detach().numpy().mean(),
            discriminator_real.cpu().detach().numpy().mean(),
            discriminator_fake.cpu().detach().numpy().mean(),
            (time.time() - time_begin_disc) / args.itd,
            time.time() - time_begin_gen,
        )
    )

    return


def validation_loop(
    val_dataloader,
    gen_optim,
    generator,
    discriminator,
    feature_extractor,
    feature_preprocess,
    criterion_mse,
    criterion_bce,
    rank,
    args,
    epoch,
):
    """Validates the model."""

    for v_ge, val_data in enumerate(val_dataloader, 0):

        X, Y = val_data
        X = torch.unsqueeze(X, dim=1).cuda(rank, non_blocking=True)
        Y = torch.unsqueeze(Y, dim=1).cuda(rank, non_blocking=True)

        gen_optim.zero_grad()
        X = generator(X)  # Generator works

        # Calculate loss
        loss_mse = utils.mean_squared_error(X, Y, criterion_mse)
        loss_adv = utils.adversarial_loss(discriminator(X), criterion_bce)

        # RSD: Now feature extractor loss
        # RSD: Same changes as above necessary.

        perc_loss = 0

        perc_loss += utils.perceptual_loss(
            feature_extractor,
            utils.perc_indexer_x,
            feature_preprocess,
            X,
            Y,
            X.shape[2],
        )
        perc_loss += utils.perceptual_loss(
            feature_extractor,
            utils.perc_indexer_y,
            feature_preprocess,
            X,
            Y,
            X.shape[3],
        )
        perc_loss += utils.perceptual_loss(
            feature_extractor,
            utils.perc_indexer_z,
            feature_preprocess,
            X,
            Y,
            X.shape[4],
        )

        generator_loss = (
            args.lmse * loss_mse + args.ladv * loss_adv + args.lperc * perc_loss
        )

        validation_loss += generator_loss.cpu().detach().numpy().mean()

    print(
        "\n[Info] Epoch: %05d, Validation loss: %.2f \n"
        % (epoch, validation_loss / len(val_dataloader))
    )
    return X


def save_checkpoint(X, args, epoch, itr_out_dir, generator, val_dataloader):
    """Saves the model and some images."""
    if epoch % args.saveiter == 0 and generator.gpu_id == 0:

        Xs = np.squeeze(X[0].cpu().detach().numpy())
        logging.info("Xs max: %s" % str(Xs.max()))
        logging.info("Xs min: %s" % str(Xs.min()))

        # output = generator(Xs)
        # output = output.cpu().detach().numpy()
        # output = np.squeeze(output)

        slice = Xs.shape[-1] // 2

        utils.save2img(Xs[slice, :, :], "%s/it%05d_x.png" % (itr_out_dir, epoch))
        utils.save2img(Xs[:, slice, :], "%s/it%05d_y.png" % (itr_out_dir, epoch))
        utils.save2img(Xs[:, :, slice], "%s/it%05d_z.png" % (itr_out_dir, epoch))

        # RSD: Generator.module because distributed.
        torch.save(
            generator.module.state_dict(), "%s/it%05d_gen.pth" % (itr_out_dir, epoch)
        )

        del Xs

    # Save model
    if epoch == 0 and generator.gpu_id == 0:

        X, Y = next(iter(val_dataloader))
        Y = np.squeeze(Y[0].cpu().detach().numpy())
        logging.debug(f"Y min {Y.min()} max: {Y.max()}")
        slice = Y.shape[0] // 2
        utils.save2img(Y[slice, :, :], "%s/gt%05d_x.png" % (itr_out_dir, epoch))
        utils.save2img(Y[:, slice, :], "%s/gt%05d_y.png" % (itr_out_dir, epoch))
        utils.save2img(Y[:, :, slice], "%s/gt%05d_z.png" % (itr_out_dir, epoch))

        X = np.squeeze(X[0].cpu().detach().numpy())
        logging.debug(f"X min {X.min()} max: {X.max()}")
        slice = Y.shape[0] // 2
        utils.save2img(X[slice, :, :], "%s/in%05d_x.png" % (itr_out_dir, epoch))
        utils.save2img(X[:, slice, :], "%s/in%05d_y.png" % (itr_out_dir, epoch))
        utils.save2img(X[:, :, slice], "%s/in%05d_z.png" % (itr_out_dir, epoch))
