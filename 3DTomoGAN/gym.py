import torch
from torch import nn as nn
import sys, os, argparse, shutil
from models import Discriminator3DTomoGAN, Generator3DTomoGAN
from data import *
import torchvision.models as models
import utils

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
    "psz",
    type=int,
    default=256,
    help="cropping patch size. If psz=0, then random cropping will be used",
)
parser.add_argument("-mbsz", type=int, default=16, help="mini-batch size")
parser.add_argument("-itg", type=int, default=1, help="iterations for G")
parser.add_argument("-itd", type=int, default=4, help="iterations for D")
parser.add_argument("-maxiter", type=int, default=8000, help="maximum number of epochs")
parser.add_argument("-dsfn", type=str, required=True, help="h5 dataset file")
# parser.add_argument("-print", type=int, default=False, help="1: print to terminal; 0: redirect to file")
parser.add_argument("-lrateg", type=float, default=1e-4, help="learning rate generator")
parser.add_argument(
    "-lrated", type=float, default=1e-4, help="learning rate discriminator"
)
parser.add_argument(
    "hparams file", type=str, default="0", help="Name of hparams file. '0' for default"
)
parser.add_argument(
    "saveiter", type=int, default=250, help="save model every saveiter iterations"
)

args, unparsed = parser.parse_known_args()

args, unparsed = parser.parse_known_args()
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

# load data

dataloader = None  # RSD.

# load models

generator = Generator3DTomoGAN()
discriminator = Discriminator3DTomoGAN()

pretrained_weights = models.VGG19_BN_Weights.IMAGENET1K_V1
preprocess = models.vgg19_bn.preprocess_input  # RSD: Check this.
preprocess = pretrained_weights.transforms()  # RSD: Check this. (transforms?)

feature_extractor = models.vgg19_bn(
    weights=pretrained_weights, include_top=False
).features  # Features are the convolutional layers of the VGG19 network Batch normalised. Not sure if better.
feature_extractor.eval()

# Define optimizers

gen_optim = torch.optim.Adam(generator.parameters(), lr=args.lrateg)
disc_optim = torch.optim.Adam(discriminator.parameters(), lr=args.lrated)

# Iterate over epochs

for epoch in range(args.maxiter + 1):
    time_git_st = time.time()
    gen_optim.zero_grad()
    discriminator.eval()
    generator.train()
    generator.requires_grad_(True)  # RSD: Necessary?

    for _ge, data in enumerate(dataloader, 0):
        # Train Generator
        # Generate fake images
        # Calculate loss
        # Backpropagate
        # Clip weights
        # Update weights

        X, Y = None, None  # dataloader.get_batch(args.mbsz, args.psz) #Get batch data

        # Train Generator
        gen_optim.zero_grad()
        generator_output = generator(X)  # Generator works
        # Calculate loss
        loss_mse = utils.mean_squared_error(generator_output, Y)
        loss_adv = utils.adversarial_loss(discriminator(generator_output))

        # RSD: Now feature extractor loss

        Y_vgg = preprocess(Y)
        generator_output_vgg = preprocess(generator_output)

        perc_loss = 0

        # RSD: Bottleneck. What about vectorised operations?
        # RSD: However, GPU already...?
        for i in range(Y_vgg.shape[-1]):  # RSD: Loop over cubic dimensions

            perc_loss += utils.feature_extraction_iteration_loss(
                feature_extractor, generator_output_vgg, Y_vgg, i
            )

        # loss_logcosh = utils.logcosh_loss(generator_output, Y) #?

        generator_loss = (
            args.lmse * loss_mse + args.ladv * loss_adv + args.lperc * perc_loss
        )
        # + args.llogcosh * loss_logcosh

        generator_loss.backward()
        gen_optim.step()

    itr_prints_gen = (
        "[Info] Epoch: %05d, %.2fs/it, gloss: %.2f (mse%.3f, adv%.3f, perc:%.3f)"
        % (
            epoch,
            (time.time() - time_git_st) / args.itg,
            generator_loss.detach().numpy().mean(),
            loss_mse.detach().numpy().mean() * args.lmse,
            loss_adv.detach().numpy().mean() * args.ladv,
            perc_loss.detach().numpy().mean() * args.lperc,
        )
    )
    time_dit_st = time.time()

    # Train Discriminator
    discriminator.train()
    discriminator.requires_grad_(True)
    gen_optim.zero_grad()
    generator.requires_grad_(False)
    generator.eval()
    for _de in range(args.itd):
        X, Y = None, None

        disc_optim.zero_grad()

        generator_output = generator(X)
        discriminator_real = discriminator(Y)
        discriminator_fake = discriminator(generator_output)

        discriminator_loss = utils.discriminator_loss(
            discriminator_real, discriminator_fake
        )

        discriminator_loss.backward()
        disc_optim.step()

    print(
        "%s; dloss: %.2f (r%.3f, f%.3f), disc_elapse: %.2fs/itr, gan_elapse: %.2fs/itr"
        % (
            itr_prints_gen,
            discriminator_loss,
            discriminator_real.detach().numpy().mean(),
            discriminator_fake.detach().numpy().mean(),
            (time.time() - time_dit_st) / args.itd,
            time.time() - time_git_st,
        )
    )

    # Save model
    if epoch % args.saveiter == 0:

        Xs, Ys = None, None

        output = generator(Xs)
        output = output.detach().cpu().numpy()
        output = np.squeeze(output)

        slice = output.shape[0] // 2

        utils.save2img(output[slice, :, :], "%s/it%05d_x" % (itr_out_dir, epoch))
        utils.save2img(output[:, slice, :], "%s/it%05d_y" % (itr_out_dir, epoch))
        utils.save2img(output[:, :, slice], "%s/it%05d_z" % (itr_out_dir, epoch))

        torch.save(generator.state_dict(), "%s/it%05d_gen.pth" % (itr_out_dir, epoch))

        # RSD: To apply saved generator:
        # generator = Generator3DTomoGAN()
        # generator.load_state_dict(torch.load('generator.pth'))
        # generator.eval()
