import torch
from torch import nn as nn
import sys, os, argparse, shutil
from models import Discriminator3DTomoGAN, Generator3DTomoGAN, TransferredResnet
from data import *
import torchvision.models as models
import utils
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# from ignite.metrics import SSIM, PSNR
import numpy as np
import logging

# RSD: Profiling
from torch.profiler import profile, record_function, ProfilerActivity, schedule

parser = argparse.ArgumentParser(
    description="3DTomoGAN, undersampled reconstruction enhancement 4D-CT"
)
parser.add_argument("-gpus", type=str, default="0", help="list of visiable GPUs")
parser.add_argument("-expName", type=str, default="debug", help="Experiment name")
parser.add_argument("-lmse", type=float, default=0.35, help="lambda mse")
parser.add_argument("-ladv", type=float, default=20, help="lambda adv")
parser.add_argument("-lperc", type=float, default=2, help="lambda perceptual")
parser.add_argument("-llogcosh", type=float, default=3, help="lambda logcosh")
parser.add_argument(
    "-psz",
    type=int,
    default=256,
    help="cropping patch size. If psz=0, then random cropping will be used",
)
parser.add_argument("-mbsz", type=int, default=16, help="mini-batch size")
parser.add_argument("-itg", type=int, default=1, help="iterations for G")
parser.add_argument("-itd", type=int, default=4, help="iterations for D")
parser.add_argument("-maxiter", type=int, default=8000, help="maximum number of epochs")
parser.add_argument("-dsfn", type=str, required=True, help="h5 dataset file")
parser.add_argument("-dsfolder", type=str, required=True, help="path dataset folder")
parser.add_argument(
    "-print", type=int, default=False, help="1: print to terminal; 0: redirect to file"
)
parser.add_argument(
    "-lrateg", type=float, default=1e-4, required=False, help="learning rate generator"
)
parser.add_argument(
    "-lrated", type=float, default=1e-4, help="learning rate discriminator"
)
parser.add_argument(
    "-hparams_file", type=str, default="0", help="Name of hparams file. '0' for default"
)  # RSD: Figure this out.
parser.add_argument(
    "-saveiter", type=int, default=250, help="save model every saveiter iterations"
)

args, unparsed = parser.parse_known_args()

logger = logging.getLogger()
logger.setLevel(logging.INFO)


if len(unparsed) > 0:
    print(
        "Unrecognized argument(s): \n%s \nProgram exiting ... ... "
        % "\n".join(unparsed)
    )
    exit(0)

if len(args.gpus) > 0:
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # disable printing INFO, WARNING, and ERROR

itr_out_dir = args.expName + "-itrOut"
if os.path.isdir(itr_out_dir):
    shutil.rmtree(itr_out_dir)
os.mkdir(itr_out_dir)  # to save temp output

# redirect print to a file
if args.print == 0:
    sys.stdout = open("%s/%s" % (itr_out_dir, "iter-prints.log"), "w")


# Devices

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)  # RSD: For one gpu
# RSD: more work if more gpus.
# device = torch.device("cpu")  # RSD: Because of memory.

logging.info("Device: %s" % device)

# RSD: Load hparams file

if args.hparams_file == "0":
    hparams = None  # {}  # RSD: Fix this
    data_hparams = {"train_split": 0.75, "val_split": 0.25, "test_split": 0.0}
else:
    hparams = torch.load(args.hparams_file)
    # RSD: Need some fixing.

# load data
mem = torch.cuda.mem_get_info()
logging.info("Before loading: " + str((mem[1] - mem[0]) / 1024 / 1024 / 1024))

full_dataset = Dataset3D(args.dsfn, args.dsfolder)  # RSD: Include hparams. First test.


# RSD: Split data into train, val, test
train_size = int(data_hparams["train_split"] * len(full_dataset))
# train_size = 6
val_size = int(data_hparams["val_split"] * len(full_dataset))
# val_size = 1
logging.info(f"Train size:   {str(train_size)} Val size:  {str(val_size)}")
test_size = len(full_dataset) - train_size - val_size
train_set, val_set, test_set = torch.utils.data.random_split(
    full_dataset, [train_size, val_size, test_size]
)

mem = torch.cuda.mem_get_info()
logging.info("After Splitting: " + str((mem[1] - mem[0]) / 1024 / 1024 / 1024))

# RSD: Temp reduction of data size due to debugging.

# RSD: Enable shuffling later.
train_dataloader = DataLoader(train_set, batch_size=args.mbsz, shuffle=False)
val_dataloader = DataLoader(val_set, batch_size=val_size, shuffle=False)
# RSD: Should save test_set to file for testing later.
torch.save(test_set, rf"{itr_out_dir}\test_set.pt")

mem = torch.cuda.mem_get_info()
logging.info("After loading: " + str((mem[1] - mem[0]) / 1024 / 1024 / 1024))

# load models

generator = Generator3DTomoGAN().to(device)
discriminator = Discriminator3DTomoGAN().to(device)

pretrained_weights = (
    models.ResNet50_Weights.IMAGENET1K_V1
)  # models.VGG19_BN_Weights.IMAGENET1K_V1
preprocess = pretrained_weights.transforms()  # RSD: Check this. (transforms?)

# feature_extractor = models.resnet152(
#     weights=pretrained_weights
# )#.features  # models.vgg19_bn(weights=pretrained_weights).features
feature_extractor = TransferredResnet(pretrained_weights)
feature_extractor.to(device)
feature_extractor.eval()
feature_extractor.requires_grad_(False)
# Features are the convolutional layers of the VGG19 network Batch normalised. Not sure if better.
# Ensure that memory is not used on the feature extractor.

mem = torch.cuda.mem_get_info()
logging.info("Loaded perc: " + str((mem[1] - mem[0]) / 1024 / 1024 / 1024))

# Define optimizers

gen_optim = torch.optim.Adam(generator.parameters(), lr=args.lrateg)
disc_optim = torch.optim.Adam(discriminator.parameters(), lr=args.lrated)

x_axis = []
train_loss = []
val_loss = []


for epoch in range(args.maxiter + 1):

    logging.info("\nEpoch: %d" % epoch)
    mem = torch.cuda.mem_get_info()
    logging.info("Begin epoch: " + str((mem[1] - mem[0]) / 1024 / 1024 / 1024))

    training_iter = 0

    assert (
        len(train_dataloader) >= args.itg + args.itd
    ), "Not enough data to perform desired epoch."

    while training_iter + args.itg + args.itd <= len(train_dataloader):

        time_git_st = time.time()

        gen_optim.zero_grad()
        discriminator.eval()
        discriminator.requires_grad_(False)
        generator.train()
        generator.requires_grad_(True)

        for _ge in range(args.itg):

            # RSD: Notice that iterations generator and discriminator are not utilised in this version. Complete the dataloader instead. Will have to check if this works.
            # for _ge, data in enumerate(train_dataloader, 0):
            # Train Generator
            # Generate fake images
            # Calculate loss
            # Backpropagate
            # Clip weights
            # Update weights

            training_iter += 1

            X, Y = next(iter(train_dataloader))  # RSD: Check if correct unpacking.
            X, Y = torch.unsqueeze(X, dim=1).to(device), torch.unsqueeze(Y, dim=1).to(
                device
            )

            # Train Generator
            gen_optim.zero_grad()
            X = generator(X)  # Generator works
            # Calculate loss
            loss_mse = utils.mean_squared_error(X, Y)

            # logging.debug("Shape consistency: %s" % str(X.shape == Y.shape))
            # logging.info("Loss MSE: %f" % loss_mse)

            # with torch.no_grad():
            #     loss_adv = utils.adversarial_loss(discriminator(X))
            loss_adv = utils.adversarial_loss(discriminator(X))
            # logging.info("Loss adv: %f" % loss_adv)

            # mem = torch.cuda.mem_get_info()
            # logging.debug("Before perc: " + str((mem[1] - mem[0]) / 1024 / 1024 / 1024))

            perc_loss = torch.zeros(1).to(device)  # 0

            # perc_loss += utils.perc_slice_loop(
            #     feature_extractor,
            #     indexer=utils.perc_indexer_x,
            #     preprocess=preprocess,
            #     X=X,
            #     Y=Y,
            #     slices=X.shape[2],
            # )
            # perc_loss += utils.perc_slice_loop(
            #     feature_extractor,
            #     indexer=utils.perc_indexer_y,
            #     preprocess=preprocess,
            #     X=X,
            #     Y=Y,
            #     slices=X.shape[3],
            # )
            # perc_loss += utils.perc_slice_loop(
            #     feature_extractor,
            #     indexer=utils.perc_indexer_z,
            #     preprocess=preprocess,
            #     X=X,
            #     Y=Y,
            #     slices=X.shape[4],
            # )

            # RSD: To ensure that no computational graph is created.
            # with torch.no_grad():
            # with torch.no_grad():
            # if True:
            # for i in range(Y.shape[2]):
            #     # RSD: Check if squeeze fucks up with minibatch size 1...? Or it works, but will size 2 be accepted? 2 not accepted.
            #     # Possibly perform this for one slice at a time since loop already. Save memory.

            #     Y_vgg = torch.squeeze(
            #         torch.stack(
            #             [Y[:, :, i, :, :], Y[:, :, i, :, :], Y[:, :, i, :, :]],
            #             dim=1,
            #         )
            #     )
            #     Y_vgg = torch.squeeze(Y_vgg)
            #     Y_vgg = preprocess(Y_vgg)

            #     X_vgg = torch.squeeze(
            #         torch.stack(
            #             [
            #                 X[:, :, i, :, :],
            #                 X[:, :, i, :, :],
            #                 X[:, :, i, :, :],
            #             ],
            #             dim=1,
            #         )
            #     )
            #     X_vgg = preprocess(X_vgg)
            #     # perc_loss += utils.feature_extraction_iteration_loss(
            #     #     feature_extractor, X_vgg, Y_vgg, i
            #     # )
            #     perc_loss += utils.slice_feature_extraction_loss(
            #         feature_extractor, X_vgg, Y_vgg
            #     )
            # mem = torch.cuda.mem_get_info()
            # logging.debug("After perc: " + str((mem[1] - mem[0]) / 1024 / 1024 / 1024))

            generator_loss = (
                args.lmse * loss_mse + args.ladv * loss_adv + args.lperc * perc_loss
            )
            logging.info(
                f"Generator loss: {generator_loss.cpu().detach().numpy()} MSE: {loss_mse.cpu().detach().numpy()} Adv: {loss_adv.cpu().detach().numpy()} Perc: {perc_loss.cpu().detach().numpy()}"
            )

            generator_loss.backward()
            gen_optim.step()

        # del X_vgg, Y_vgg

        itr_prints_gen = (
            "[Info] Epoch: %05d, %.2fs/it, gloss: %.2f (mse%.3f, adv%.3f, perc:%.3f)"
            % (
                epoch,
                (time.time() - time_git_st) / args.itg,
                generator_loss.cpu().detach().numpy(),
                loss_mse.cpu().detach().numpy().mean() * args.lmse,
                loss_adv.cpu().detach().numpy().mean() * args.ladv,
                0,  # perc_loss.cpu().detach().numpy().mean() * args.lperc,
            )
        )

        train_loss.append(generator_loss.cpu().detach().numpy())
        time_dit_st = time.time()

        # Train Discriminator
        discriminator.train()
        discriminator.requires_grad_(True)
        gen_optim.zero_grad()
        generator.eval()
        generator.requires_grad_(False)

        mem = torch.cuda.mem_get_info()
        logging.debug(
            "Before discriminator: " + str((mem[1] - mem[0]) / 1024 / 1024 / 1024)
        )

        # for _de, data in enumerate(train_dataloader, 0):
        for _de in range(args.itd):

            training_iter += 1
            # X, Y = data
            X, Y = next(iter(train_dataloader))
            X, Y = torch.unsqueeze(X, dim=1).to(device), torch.unsqueeze(Y, dim=1).to(
                device
            )

            disc_optim.zero_grad()
            # with torch.no_grad():
            X = generator(X)

            # logging.info(f"Req_grad {X.requires_grad} {Y.requires_grad}")

            discriminator_real = discriminator(Y)
            discriminator_fake = discriminator(X)
            # RSD: Does this work? Same loss every time.
            discriminator_loss = utils.discriminator_loss(
                discriminator_real, discriminator_fake
            )
            logging.info("Discriminator loss: %f" % discriminator_loss)

            # logging.info(f"Req grad {discriminator_loss.requires_grad}")

            discriminator_loss.backward()

            logging.info(
                f"Discriminator grad:{ [discriminator.layers[i].weight.grad.max().cpu().detach().numpy() for i in range(0, 7, 2)]}"
            )
            disc_optim.step()

        disc_optim.zero_grad()

        print(
            "%s; dloss: %.2f (r%.3f, f%.3f), disc_elapse: %.2fs/itr, gan_elapse: %.2fs/itr"
            % (
                itr_prints_gen,
                discriminator_loss.cpu().detach().numpy().mean(),
                discriminator_real.cpu().detach().numpy().mean(),
                discriminator_fake.cpu().detach().numpy().mean(),
                (time.time() - time_dit_st) / args.itd,
                time.time() - time_git_st,
            )
        )
    mem = torch.cuda.mem_get_info()
    logging.debug("Before validation: " + str((mem[1] - mem[0]) / 1024 / 1024 / 1024))

    # Validation
    generator.eval()
    generator.requires_grad_(False)
    validation_loss = 0
    ssim_loss = 0
    psnr_loss = 0
    # ssim = SSIM(data_range=1.0)
    # psnr = PSNR(data_range=1.0)

    with torch.no_grad():
        for v_ge, val_data in enumerate(val_dataloader, 0):

            X, Y = val_data
            X_save = X.clone()
            X, Y = torch.unsqueeze(X, dim=1).to(device), torch.unsqueeze(Y, dim=1).to(
                device
            )

            gen_optim.zero_grad()
            X = generator(X)  # Generator works

            # Calculate loss
            loss_mse = utils.mean_squared_error(X, Y)
            loss_adv = utils.adversarial_loss(discriminator(X))

            # RSD: Now feature extractor loss
            # RSD: Same changes as above necessary.

            perc_loss = torch.zeros(1).to(device)

            # perc_loss += utils.perc_slice_loop(
            #     feature_extractor,
            #     indexer=utils.perc_indexer_x,
            #     preprocess=preprocess,
            #     X=X,
            #     Y=Y,
            #     slices=X.shape[2],
            # )
            # perc_loss += utils.perc_slice_loop(
            #     feature_extractor,
            #     indexer=utils.perc_indexer_y,
            #     preprocess=preprocess,
            #     X=X,
            #     Y=Y,
            #     slices=X.shape[3],
            # )
            # perc_loss += utils.perc_slice_loop(
            #     feature_extractor,
            #     indexer=utils.perc_indexer_z,
            #     preprocess=preprocess,
            #     X=X,
            #     Y=Y,
            #     slices=X.shape[4],
            # )

            generator_loss = (
                args.lmse * loss_mse + args.ladv * loss_adv + args.lperc * perc_loss
            )

            logging.info(
                "Validation loss: %f, mse: %f, adv: %f, perc: %f"
                % (
                    generator_loss.cpu().detach().numpy().mean(),
                    loss_mse.cpu().detach().numpy().mean(),
                    loss_adv.cpu().detach().numpy().mean(),
                    perc_loss.cpu().detach().numpy().mean(),
                )
            )

            validation_loss += generator_loss.cpu().detach().numpy().mean()

    print(
        "\n[Info] Epoch: %05d, Validation loss: %.2f \n"
        % (epoch, validation_loss / len(val_dataloader))
    )  # RSD ++ Accuracy or PSNR or SSIM
    x_axis.append(len(train_loss))
    val_loss.append(validation_loss / len(val_dataloader))

    with torch.no_grad():
        if epoch % args.saveiter == 0:

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

            plt.imshow(Xs[slice, :, :])
            plt.colorbar()
            plt.savefig("%s/it%05d_x_plot.png" % (itr_out_dir, epoch))
            plt.close()

            torch.save(
                generator.state_dict(), "%s/it%05d_gen.pth" % (itr_out_dir, epoch)
            )

            del Xs

        # Save model
        if epoch % args.saveiter == 0:

            # X, Y = next(iter(val_dataloader))
            Y = np.squeeze(Y[0].cpu().detach().numpy())
            logging.debug(f"Y min {Y.min()} max: {Y.max()}")
            slice = Y.shape[0] // 2
            utils.save2img(Y[slice, :, :], "%s/gt%05d_x.png" % (itr_out_dir, epoch))
            utils.save2img(Y[:, slice, :], "%s/gt%05d_y.png" % (itr_out_dir, epoch))
            utils.save2img(Y[:, :, slice], "%s/gt%05d_z.png" % (itr_out_dir, epoch))

            X = np.squeeze(X_save[0].cpu().detach().numpy())
            logging.debug(f"X min {X.min()} max: {X.max()}")
            slice = Y.shape[0] // 2
            utils.save2img(X[slice, :, :], "%s/in%05d_x.png" % (itr_out_dir, epoch))
            utils.save2img(X[:, slice, :], "%s/in%05d_y.png" % (itr_out_dir, epoch))
            utils.save2img(X[:, :, slice], "%s/in%05d_z.png" % (itr_out_dir, epoch))

            del X, Y  # RSD: Necessary?

        # RSD: To apply saved generator:
        # generator = Generator3DTomoGAN()
        # generator.load_state_dict(torch.load('generator.pth'))
        # generator.eval()

    # p.step()  # Profiling

    # Save loss curves
    # Need to plot mse as well.

    plt.plot(train_loss, label="train")
    plt.plot(x_axis, val_loss, label="validation")
    plt.legend()
    plt.savefig("%s/loss.png" % (itr_out_dir))
    plt.close()
