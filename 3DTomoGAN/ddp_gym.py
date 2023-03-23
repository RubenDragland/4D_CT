import torch
from torch import nn as nn
import sys, os, argparse, shutil
from models import Discriminator3DTomoGAN, Generator3DTomoGAN, TransferredResnet
import torchvision.models as models
from data import *
import torchvision.models as models
import utils
import ddp_utils
from torch.utils.data import DataLoader

# from ignite.metrics import SSIM, PSNR
import numpy as np
import logging

# RSD: Profiling
from parser_gym import parser_args_gym

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


def multiprocess_train_code(
    rank,
    num_gpus,
    train_set,
    val_set,
    hparams,
    data_hparams,
    args,  # RSD: Keep args from parser separate or in hparams?
    itr_out_dir,
):
    """DDP training code. Distributes data and models across gpus. Performs processes."""

    torch.distributed.init_process_group(
        backend="nccl",
        rank=rank,
        world_size=num_gpus,
        init_method="env://",
    )  # RSD: Hope init_method is correct

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_set)
    train_dataloader = DataLoader(
        dataset=train_set,
        batch_size=args.mbsz,
        shuffle=False,
        num_workers=4 * num_gpus,
        pin_memory=True,  # RSD? Believe this is performance > memory
        sampler=train_sampler,
    )

    val_sampler = torch.utils.data.distributed.DistributedSampler(val_set)
    val_dataloader = DataLoader(
        dataset=val_set,
        batch_size=args.mbsz,
        shuffle=False,
        num_workers=4 * num_gpus,
        pin_memory=True,  # RSD? Believe this is performance > memory
        sampler=val_sampler,
    )

    logging.debug("Rank: " + str(rank) + " - Data loaded.")

    # Inits models across the gpus

    generator = Generator3DTomoGAN().cuda()
    discriminator = Discriminator3DTomoGAN().cuda()

    generator = ddp_utils.init_model(generator, rank)
    discriminator = ddp_utils.init_model(discriminator, rank)

    pretrained_weights = (
        models.ResNet50_Weights.IMAGENET1K_V1
    )  # models.VGG19_BN_Weights.IMAGENET1K_V1  # RSD: Change this
    preprocess = pretrained_weights.transforms()  # RSD: Check this. (transforms?)

    feature_extractor = TransferredResnet(
        pretrained_weights
    ).cuda()  # models.vgg19_bn(weights=pretrained_weights).features
    # feature_extractor = ddp_utils.init_model(feature_extractor, rank)
    # feature_extractor = nn.SyncBatchNorm.convert_sync_batchnorm(
    #     feature_extractor
    # )  # RSD: Believe this is necessary

    # Inits the optimizers
    gen_optim = torch.optim.Adam(generator.parameters(), lr=args.lrateg)
    disc_optim = torch.optim.Adam(discriminator.parameters(), lr=args.lrated)

    # Distributes the criterions across gpus
    binary_cross_entropy = nn.BCEWithLogitsLoss()
    binary_cross_entropy.cuda(rank)
    mean_squared_error = nn.MSELoss()
    mean_squared_error.cuda(rank)

    assert args.itg + args.itd > len(
        train_dataloader
    ), "Must be enough data for each epoch"

    for epoch in range(args.maxiter + 1):
        train_sampler.set_epoch(epoch)
        val_sampler.set_epoch(epoch)

        training_iter = 0

        while training_iter + args.itg + args.itd < len(train_dataloader):

            gen_optim.zero_grad()
            discriminator.eval()
            discriminator.requires_grad_(False)
            generator.train()
            generator.requires_grad_(True)

            time_git_st = time.time()

            itr_prints_gen = ddp_utils.generator_loop(
                train_dataloader,
                gen_optim,
                generator,
                discriminator,
                feature_extractor,
                preprocess,
                mean_squared_error,
                binary_cross_entropy,
                rank,
                args,
                epoch,
                time_begin=time_git_st,
            )

            time_dit_st = time.time()

            # Train Discriminator
            discriminator.train()
            discriminator.requires_grad_(True)
            gen_optim.zero_grad()
            generator.eval()
            generator.requires_grad_(False)

            ddp_utils.discriminator_loop(
                train_dataloader,
                disc_optim,
                generator,
                discriminator,
                binary_cross_entropy,
                rank,
                args,
                epoch,
                time_begin_disc=time_dit_st,
                time_begin_gen=time_git_st,
                itr_prints_gen=itr_prints_gen,
            )

            generator.eval()
            generator.requires_grad_(False)

            with torch.no_grad():
                X = ddp_utils.validation_loop(
                    val_dataloader,
                    gen_optim,
                    generator,
                    discriminator,
                    feature_extractor,
                    preprocess,
                    mean_squared_error,
                    binary_cross_entropy,
                    rank,
                    args,
                    epoch,
                )

                ddp_utils.save_checkpoint(
                    X, args, epoch, itr_out_dir, generator, val_dataloader, rank
                )

    return


if __name__ == "__main__":

    parser = parser_args_gym()

    args, unparsed = parser.parse_known_args()

    if len(unparsed) > 0:
        print(
            "Unrecognized argument(s): \n%s \nProgram exiting ... ... "
            % "\n".join(unparsed)
        )
        exit(0)

    if len(args.gpus) > 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    os.environ[
        "TF_CPP_MIN_LOG_LEVEL"
    ] = "3"  # disable printing INFO, WARNING, and ERROR

    itr_out_dir = args.expName + "-itrOut"
    if os.path.isdir(itr_out_dir):
        shutil.rmtree(itr_out_dir)
    os.mkdir(itr_out_dir)  # to save temp output

    # redirect print to a file
    if args.print == 0:
        sys.stdout = open("%s/%s" % (itr_out_dir, "iter-prints.log"), "w")

    # Devices; Separates CPU form GPU case.

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_gpus = torch.cuda.device_count()  # RSD Or use parser arg

    logging.debug("Number of GPUs: " + str(num_gpus), "Device: " + str(device))

    # RSD: Load hparams file

    if args.hparams_file == "0":
        hparams = None  # {}  # RSD: Fix this
        data_hparams = {"train_split": 0.875, "val_split": 0.125, "test_split": 0.0}
    else:
        hparams = torch.load(args.hparams_file)
        # RSD: Need some fixing.

    # load data
    mem = torch.cuda.mem_get_info()
    logging.debug("Before loading: " + str((mem[1] - mem[0]) / 1024 / 1024 / 1024))

    full_dataset = Dataset3D(args.dsfn, args.dsfolder)

    # RSD: Split data into train, val, test
    train_size = int(data_hparams["train_split"] * len(full_dataset))
    train_size = 6
    val_size = int(data_hparams["val_split"] * len(full_dataset))
    val_size = 1
    test_size = len(full_dataset) - train_size - val_size
    train_set, val_set, test_set = torch.utils.data.random_split(
        full_dataset, [train_size, val_size, test_size]
    )

    # Perhaps init models here...?

    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29500"

    torch.multiprocessing.spawn(
        multiprocess_train_code,
        args=(num_gpus, train_set, val_set, hparams, data_hparams, args, itr_out_dir),
        nprocs=num_gpus,
        join=True,
    )  # RSD: Unsure about join
