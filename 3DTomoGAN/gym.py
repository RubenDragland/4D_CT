import torch
from torch import nn as nn
import sys, os, argparse, shutil
from models import Discriminator3DTomoGAN, Generator3DTomoGAN
from data import *
import torchvision.models as models
import utils
from torch.utils.data import DataLoader

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
logger.setLevel(logging.DEBUG)


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

logging.debug("Device: %s" % device)

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
train_size = 1
val_size = int(data_hparams["val_split"] * len(full_dataset))
val_size = 1
test_size = len(full_dataset) - train_size - val_size
train_set, val_set, test_set = torch.utils.data.random_split(
    full_dataset, [train_size, val_size, test_size]
)

# RSD: Temp reduction of data size due to debugging.

train_dataloader = DataLoader(train_set, batch_size=args.mbsz, shuffle=True)
val_dataloader = DataLoader(val_set, batch_size=val_size, shuffle=True)
# RSD: Should save test_set to file for testing later.
torch.save(test_set, rf"{itr_out_dir}\test_set.pt")

mem = torch.cuda.mem_get_info()
logging.debug("After loading: " + str((mem[1] - mem[0]) / 1024 / 1024 / 1024))

# load models

generator = Generator3DTomoGAN().to(device)
discriminator = Discriminator3DTomoGAN().to(device)

pretrained_weights = models.VGG19_BN_Weights.IMAGENET1K_V1
preprocess = pretrained_weights.transforms()  # RSD: Check this. (transforms?)

feature_extractor = models.vgg19_bn(weights=pretrained_weights).features
feature_extractor.eval().to(device)
# Features are the convolutional layers of the VGG19 network Batch normalised. Not sure if better.

mem = torch.cuda.mem_get_info()
logging.debug("Loaded vgg: " + str((mem[1] - mem[0]) / 1024 / 1024 / 1024))

# Define optimizers

gen_optim = torch.optim.Adam(generator.parameters(), lr=args.lrateg)
disc_optim = torch.optim.Adam(discriminator.parameters(), lr=args.lrated)

# Iterate over epochs

# RSD: Start profiling
def handler(p):
    # print(p.key_averages().table(sort_by="self_cpu_time_total", row_limit=10))
    print(
        p.key_averages(group_by_input_shape=True).table(
            sort_by="cpu_time_total", row_limit=10
        )
    )
    print(
        p.key_averages(group_by_input_shape=True).table(
            sort_by="self_cpu_memory_usage", row_limit=10
        )
    )
    return


# my_schedule = schedule(skip_first=1, wait=0, warmup=1, active=5, repeat=2)

# with profile(
#     activities=[ProfilerActivity.CPU],
#     profile_memory=True,
#     record_shapes=True,
#     schedule=my_schedule,
#     on_trace_ready=handler,
# ) as p:

for epoch in range(args.maxiter + 1):

    logging.debug("Epoch: %d" % epoch)

    time_git_st = time.time()
    gen_optim.zero_grad()
    discriminator.eval()
    generator.train()
    generator.requires_grad_(True)  # RSD: Necessary?

    mem = torch.cuda.mem_get_info()
    logging.debug("Begin epoch: " + str((mem[1] - mem[0]) / 1024 / 1024 / 1024))

    # RSD: Notice that iterations generator and discriminator are not utilised in this version. Complete the dataloader instead. Will have to check if this works.
    for _ge, data in enumerate(train_dataloader, 0):
        # Train Generator
        # Generate fake images
        # Calculate loss
        # Backpropagate
        # Clip weights
        # Update weights

        logging.debug(f"Gen Batch: {_ge}")

        X, Y = data  # RSD: Check if correct unpacking.
        logging.debug(f"Dataloader Dtype:   {X.dtype}")
        X, Y = torch.unsqueeze(X, dim=1).to(device), torch.unsqueeze(Y, dim=1).to(
            device
        )
        logging.debug(f"Dtype:   {X.dtype}")

        logging.debug("X shape: %s" % str(X.shape))
        logging.debug("Y shape: %s" % str(Y.shape))

        # Train Generator
        gen_optim.zero_grad()
        generator_output = generator(X)  # Generator works
        # Calculate loss
        loss_mse = utils.mean_squared_error(generator_output, Y)

        logging.debug("Generator output shape: %s" % str(generator_output.shape))
        logging.debug("Loss MSE: %f" % loss_mse)
        loss_adv = utils.adversarial_loss(discriminator(generator_output))
        logging.debug("Loss adv: %f" % loss_adv)

        # RSD: Now feature extractor loss

        # Y_vgg = preprocess(Y)
        # generator_output_vgg = preprocess(generator_output)
        # RSD: Prepare vgg net shape.
        # Y_vgg = torch.zeros((Y.shape[0], 3, Y.shape[2], 224, 224)).to(device)
        # generator_output_vgg = torch.zeros((Y.shape[0], 3, Y.shape[2], 224, 224)).to(
        #     device
        # )
        # RSD: Check requires grad etc.

        mem = torch.cuda.mem_get_info()
        logging.debug("Before perc: " + str((mem[1] - mem[0]) / 1024 / 1024 / 1024))

        perc_loss = 0

        for i in range(Y.shape[2]):
            # RSD: Check if squeeze fucks up with minibatch size 1...? Or it works, but will size 2 be accepted? 2 not accepted.
            # Possibly perform this for one slice at a time since loop already. Save memory.

            Y_vgg = torch.squeeze(
                torch.stack(
                    [Y[:, :, i, :, :], Y[:, :, i, :, :], Y[:, :, i, :, :]],
                    dim=1,
                )
            )
            Y_vgg = torch.squeeze(Y_vgg)
            Y_vgg = preprocess(Y_vgg)

            generator_output_vgg = torch.squeeze(
                torch.stack(
                    [
                        generator_output[:, :, i, :, :],
                        generator_output[:, :, i, :, :],
                        generator_output[:, :, i, :, :],
                    ],
                    dim=1,
                )
            )
            generator_output_vgg = preprocess(generator_output_vgg)
            # perc_loss += utils.feature_extraction_iteration_loss(
            #     feature_extractor, generator_output_vgg, Y_vgg, i
            # )
            perc_loss += utils.slice_feature_extraction_loss(
                feature_extractor, generator_output_vgg, Y_vgg
            )

        logging.debug("Perc loss: %f" % perc_loss)
        mem = torch.cuda.mem_get_info()
        logging.debug("After perc: " + str((mem[1] - mem[0]) / 1024 / 1024 / 1024))

        # RSD: Loop over cubic dimensions. Also, preprocessing is only done for every crossection.

        # loss_logcosh = utils.logcosh_loss(generator_output, Y) #?

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
            (time.time() - time_git_st) / args.itg,
            generator_loss.cpu().detach().numpy(),
            loss_mse.cpu().detach().numpy().mean() * args.lmse,
            loss_adv.cpu().detach().numpy().mean() * args.ladv,
            perc_loss.cpu().detach().numpy().mean() * args.lperc,
        )
    )
    time_dit_st = time.time()

    # Train Discriminator
    discriminator.train()
    discriminator.requires_grad_(True)
    gen_optim.zero_grad()
    generator.requires_grad_(False)
    generator.eval()

    mem = torch.cuda.mem_get_info()
    logging.debug(
        "Before discriminator: " + str((mem[1] - mem[0]) / 1024 / 1024 / 1024)
    )

    for _de, data in enumerate(train_dataloader, 0):
        X, Y = data
        X, Y = torch.unsqueeze(X, dim=1).to(device), torch.unsqueeze(Y, dim=1).to(
            device
        )

        disc_optim.zero_grad()

        generator_output = generator(X)
        discriminator_real = discriminator(Y)
        discriminator_fake = discriminator(generator_output)

        discriminator_loss = utils.discriminator_loss(
            discriminator_real, discriminator_fake
        )
        logging.debug("Discriminator loss: %f" % discriminator_loss)

        discriminator_loss.backward()
        disc_optim.step()

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
    for v_ge, val_data in enumerate(val_dataloader, 0):

        X, Y = val_data
        X, Y = torch.unsqueeze(X, dim=1).to(device), torch.unsqueeze(Y, dim=1).to(
            device
        )

        gen_optim.zero_grad()
        generator_output = generator(X)  # Generator works

        # Calculate loss
        loss_mse = utils.mean_squared_error(generator_output, Y)
        loss_adv = utils.adversarial_loss(discriminator(generator_output))

        # RSD: Now feature extractor loss
        # RSD: Same changes as above necessary.

        perc_loss = 0

        for i in range(Y.shape[2]):
            Y_vgg = preprocess(
                torch.squeeze(
                    torch.stack(
                        [Y[:, :, i, :, :], Y[:, :, i, :, :], Y[:, :, i, :, :]],
                        dim=1,
                    )
                )
            )
            generator_output_vgg = preprocess(
                torch.squeeze(
                    torch.stack(
                        [
                            generator_output[:, :, i, :, :],
                            generator_output[:, :, i, :, :],
                            generator_output[:, :, i, :, :],
                        ],
                        dim=1,
                    )
                )
            )

            perc_loss += utils.slice_feature_extraction_loss(
                feature_extractor, generator_output_vgg, Y_vgg
            )

        generator_loss = (
            args.lmse * loss_mse + args.ladv * loss_adv + args.lperc * perc_loss
        )

        validation_loss += generator_loss.cpu().detach().numpy().mean()

    print(
        "\n[Info] Epoch: %05d, Validation loss: %.2f \n"
        % (epoch, validation_loss / len(val_dataloader))
    )  # RSD ++ Accuracy or PSNR or SSIM

    # Save model
    if epoch == 0:
        Ys = np.squeeze(Y[0].cpu().detach().numpy())
        slice = Ys.shape[0] // 2
        utils.save2img(Ys[slice, :, :], "%s/gt%05d_x.png" % (itr_out_dir, epoch))
        utils.save2img(Ys[:, slice, :], "%s/gt%05d_y.png" % (itr_out_dir, epoch))
        utils.save2img(Ys[:, :, slice], "%s/gt%05d_z.png" % (itr_out_dir, epoch))

        Xs = np.squeeze(X[0].cpu().detach().numpy())
        slice = Y.shape[0] // 2
        utils.save2img(Xs[slice, :, :], "%s/in%05d_x.png" % (itr_out_dir, epoch))
        utils.save2img(Xs[:, slice, :], "%s/in%05d_y.png" % (itr_out_dir, epoch))
        utils.save2img(Xs[:, :, slice], "%s/in%05d_z.png" % (itr_out_dir, epoch))

    if epoch % args.saveiter == 0:

        Xs = np.squeeze(generator_output[0].cpu().detach().numpy())
        logging.debug("Xs.shape: %s" % str(Xs.shape))

        # output = generator(Xs)
        # output = output.cpu().detach().numpy()
        # output = np.squeeze(output)

        slice = Xs.shape[-1] // 2

        utils.save2img(Xs[slice, :, :], "%s/it%05d_x.png" % (itr_out_dir, epoch))
        utils.save2img(Xs[:, slice, :], "%s/it%05d_y.png" % (itr_out_dir, epoch))
        utils.save2img(Xs[:, :, slice], "%s/it%05d_z.png" % (itr_out_dir, epoch))

        torch.save(generator.state_dict(), "%s/it%05d_gen.pth" % (itr_out_dir, epoch))

        # RSD: To apply saved generator:
        # generator = Generator3DTomoGAN()
        # generator.load_state_dict(torch.load('generator.pth'))
        # generator.eval()

    # p.step()  # Profiling
