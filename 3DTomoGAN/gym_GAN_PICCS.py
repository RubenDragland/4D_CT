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
import json


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
parser.add_argument("-idx", type=int, default=0, help="index of the training sample")

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
    # os.environ["CUDA_VISIBLE_DEVICES"] = [str(i) for i in range(int(args.gpus))]
    pass
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # disable printing INFO, WARNING, and ERROR

itr_out_dir = args.expName + "-itrOut"
if os.path.isdir(itr_out_dir):
    shutil.rmtree(itr_out_dir)
os.mkdir(itr_out_dir)  # to save temp output

# redirect print to a file
if args.print == 0:
    sys.stdout = open("%s/%s" % (itr_out_dir, "iter-prints.log"), "w")


# Devices

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")  # RSD: Because of memory.

logging.info("Device: %s" % device)

# RSD: Load hparams file

if args.hparams_file == "0":
    hparams = None  # {}  # RSD: Fix this
    data_hparams = {"train_split": 0.90, "val_split": 0.10, "test_split": 0.0}
else:
    # hparams = torch.load(args.hparams_file)
    my_path = os.path.abspath(os.path.dirname(__file__))
    with open(os.path.join(my_path, "hparams", f"{args.hparams_file}.json"), "r") as d:
        hparams = json.load(d)
        data_hparams = {
            "train_split": hparams["train_split"],
            "val_split": 1 - hparams["train_split"],
            "test_split": 0.0,
        }

# load data
mem = torch.cuda.mem_get_info()
logging.info("Before loading: " + str((mem[1] - mem[0]) / 1024 / 1024 / 1024))

train_set = Dataset3D(args.dsfn, args.dsfolder, hparams)
val_set = Dataset3D(args.dsfn, args.dsfolder, hparams)


args.lrateg = hparams["lrateg"]
args.lrated = hparams["lrated"]
args.mbsz = hparams["mbsz"]
args.lmse = hparams["lmse"]
args.ladv = hparams["ladv"]
args.itg = hparams["itg"]
args.itd = hparams["itd"]
args.lperc = hparams["lperc"]


mem = torch.cuda.mem_get_info()
logging.info("After Splitting: " + str((mem[1] - mem[0]) / 1024 / 1024 / 1024))


train_dataloader = DataLoader(train_set, batch_size=args.mbsz, shuffle=True)
val_dataloader = DataLoader(val_set, batch_size=args.mbsz, shuffle=True)

mem = torch.cuda.mem_get_info()
logging.info("After loading: " + str((mem[1] - mem[0]) / 1024 / 1024 / 1024))

# load models

generator = Generator3DTomoGAN(normalise=True).to(device)
discriminator = Discriminator3DTomoGAN().to(device)


if hparams["transfer_model"] is not None:
    my_path = os.path.abspath(os.path.dirname(__file__))
    generator.load_state_dict(
        torch.load(
            os.path.join(my_path, "transfer_models", f"{hparams['transfer_model']}.pth")
        )
    )
    logging.info("Transfer loaded")


# RSD: Consider another perceptual loss again. Possibly for discussion.
pretrained_weights = (
    models.ResNet50_Weights.IMAGENET1K_V1
)  # models.VGG19_BN_Weights.IMAGENET1K_V1
preprocess = pretrained_weights.transforms()  # RSD: Check this. (transforms?)


feature_extractor = TransferredResnet(pretrained_weights)
feature_extractor.to(device)
feature_extractor.eval()
feature_extractor.requires_grad_(False)
# Ensure that memory is not used on the feature extractor.

mem = torch.cuda.mem_get_info()
logging.info("Loaded perc: " + str((mem[1] - mem[0]) / 1024 / 1024 / 1024))


# Define optimizers

gen_optim = torch.optim.Adam(generator.parameters(), lr=args.lrateg)
disc_optim = torch.optim.Adam(discriminator.parameters(), lr=args.lrated)
# RSD: Was divided by 10. Unsure why. Min learning rate used in scheduler.

# RSD: Argparse gives minimal learning rate.
gen_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    gen_optim, T_max=args.maxiter, eta_min=args.lrateg
)
disc_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    disc_optim, T_max=args.maxiter, eta_min=args.lrated
)

x_axis = []
train_loss = []
val_loss = []
mse_list = []
adv_list = []
ssim_list = []
psnr_list = []

loss_adv = torch.zeros(1)
generator_loss = torch.zeros(1)
discriminator_loss = torch.zeros(1)


for epoch in range(args.maxiter + 1):
    logging.info("\nEpoch: %d" % epoch)
    mem = torch.cuda.mem_get_info()
    logging.info("Begin epoch: " + str((mem[1] - mem[0]) / 1024 / 1024 / 1024))

    training_iter = 0

    while training_iter < len(train_dataloader):
        time_git_st = time.time()

        gen_optim.zero_grad()
        discriminator.eval()
        discriminator.requires_grad_(False)
        generator.train()
        generator.requires_grad_(True)

        for _ge in range(args.itg):
            training_iter += 1

            # X, Y = next(iter(train_dataloader))
            X, Y = train_set.__getitem__(args.idx)  # RSD: Difference between iterators?

            X, Y = torch.unsqueeze(X, dim=0).unsqueeze(0).to(device), torch.unsqueeze(
                Y, dim=0
            ).unsqueeze(0).to(device)

            # Train Generator
            gen_optim.zero_grad()
            X = generator(X)  # Generator works
            # Calculate loss
            loss_mse = utils.mean_squared_error(X, Y)

            loss_adv = utils.adversarial_loss(discriminator(X))
            # logging.info("Loss adv: %f" % loss_adv)

            # mem = torch.cuda.mem_get_info()
            # logging.debug("Before perc: " + str((mem[1] - mem[0]) / 1024 / 1024 / 1024))

            # perc_loss = torch.zeros(1).to(device)  # 0
            perc_loss = 0

            perc_loss += utils.perc_slice_loop(
                feature_extractor,
                indexer=utils.perc_indexer_x,
                preprocess=preprocess,
                X=X,
                Y=Y,
                slices=X.shape[2],
            )

            generator_loss = (
                args.lmse * loss_mse + args.ladv * loss_adv + args.lperc * perc_loss
            )
            logging.info(
                f"Generator loss: {generator_loss.cpu().detach().numpy()} MSE: {loss_mse.cpu().detach().numpy()} Adv: {loss_adv.cpu().detach().numpy()} Perc: {perc_loss.cpu().detach().numpy()}"
            )

            if generator_loss < args.lmse * args.ladv * 1000:
                generator_loss.backward()
                gen_optim.step()

            if loss_adv < 0.69:  # RSD: Needs to be tweaked.
                break

        itr_prints_gen = (
            "[Info] Epoch: %05d, %.2fs/it, gloss: %.2f (mse%.3f, adv%.3f, perc:%.3f)"
            % (
                epoch,
                (time.time() - time_git_st) / args.itg,
                generator_loss.cpu().detach().numpy(),
                loss_mse.cpu().detach().numpy().mean() * args.lmse,
                loss_adv.cpu().detach().numpy().mean() * args.ladv,
                perc_loss.cpu().detach().numpy().mean()
                * args.lperc,  # perc_loss.cpu().detach().numpy().mean() * args.lperc,
            )
        )

        train_loss.append(generator_loss.cpu().detach().numpy())
        mse_list.append(loss_mse.cpu().detach().numpy().mean() * args.lmse)
        adv_list.append(loss_adv.cpu().detach().numpy().mean() * args.ladv)
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

        # RSD: Remember to fully implement how this is supposed to work.
        for _de in range(args.itd):
            training_iter += 1
            # X, Y = next(iter(train_dataloader))
            X, Y = train_set.__getitem__(args.idx)
            X, Y = torch.unsqueeze(X, dim=0).unsqueeze(0).to(device), torch.unsqueeze(
                Y, dim=0
            ).unsqueeze(0).to(device)

            disc_optim.zero_grad()
            X = generator(X)

            discriminator_real = discriminator(Y)
            discriminator_fake = discriminator(X)

            discriminator_loss = utils.discriminator_loss(
                discriminator_real, discriminator_fake
            )
            logging.info("Discriminator loss: %f" % discriminator_loss)

            discriminator_loss.backward()

            # logging.info(
            #     f"Discriminator grad:{ [discriminator.layers[i].weight.grad.max().cpu().detach().numpy() for i in range(0, 7, 2)]}"
            # )
            disc_optim.step()

            if discriminator_loss < 1.35:
                break

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
    ssim_compare = 0
    psnr_loss = 0

    with torch.no_grad():
        for v_ge, val_data in enumerate(val_dataloader, 0):
            X, Y = val_data
            X_save = X.clone()
            X, Y = torch.unsqueeze(X, dim=0).to(device), torch.unsqueeze(Y, dim=0).to(
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
            perc_loss = 0

            perc_loss += utils.perc_slice_loop(
                feature_extractor,
                indexer=utils.perc_indexer_x,
                preprocess=preprocess,
                X=X,
                Y=Y,
                slices=X.shape[2],
            )

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
            mssim = 0
            mssimc = 0
            for i in range(X.shape[2]):
                ms, _ = utils.calc_mssim(
                    torch.squeeze(Y[:, :, i]).cpu().detach().numpy(),
                    torch.squeeze(X[:, :, i]).cpu().detach().numpy(),
                )
                mssim += ms
                ms, _ = utils.calc_mssim(
                    torch.squeeze(Y[:, :, i]).cpu().detach().numpy(),
                    torch.squeeze(X_save[:, :, i]).cpu().detach().numpy(),
                )
                mssimc += ms
            ssim_loss += mssim / int(X.shape[2])  # utils.calc_ssim(        #)
            psnr_loss += utils.calc_psnr(
                Y.cpu().detach().numpy(), X.cpu().detach().numpy()
            )
            ssim_compare += mssimc / int(X.shape[2])

    print(
        "\n[Info] Epoch: %05d, Validation loss: %.2f \n"
        % (epoch, validation_loss / len(val_dataloader))
    )
    print(
        "[Info] Epoch: %05d, Validation ssim: %.2f (%.2f)\n"
        % (epoch, ssim_loss / len(val_dataloader), ssim_compare / len(val_dataloader))
    )

    logging.info(
        f"Validation ssim: {ssim_loss/len(val_dataloader)} ({ssim_compare / len(val_dataloader)})"
    )

    x_axis.append(len(train_loss))
    val_loss.append(validation_loss / len(val_dataloader))
    ssim_list.append(ssim_loss / len(val_dataloader))
    psnr_list.append(psnr_loss / len(val_dataloader))

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

            del X, Y, X_save  # RSD: Necessary?

    # Save loss curves
    # Need to plot mse as well.

    try:
        # Save some data for discussion
        np.save("%s/train_loss.npy" % (itr_out_dir), np.array(train_loss))
        np.save("%s/val_loss.npy" % (itr_out_dir), np.array(val_loss))
        np.save("%s/x_axis.npy" % (itr_out_dir), np.array(x_axis))
        np.save("%s/mse_list.npy" % (itr_out_dir), np.array(mse_list))
        np.save("%s/adv_list.npy" % (itr_out_dir), np.array(adv_list))
        np.save(
            "%s/loss_weights.npy" % (itr_out_dir),
            np.array([args.lmse, args.ladv, args.lperc]),
        )
        np.save("%s/ssim_list.npy" % (itr_out_dir), np.array(ssim_list))
        np.save("%s/psnr_list.npy" % (itr_out_dir), np.array(psnr_list))
    except:
        pass

    finally:
        # plt.plot(train_loss, label="train")
        plt.plot(x_axis, val_loss, label="validation")
        plt.legend()
        plt.savefig("%s/loss.png" % (itr_out_dir))
        plt.close()
