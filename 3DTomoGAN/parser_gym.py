import sys, os, argparse, shutil


def parser_args_gym():

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
    parser.add_argument(
        "-maxiter", type=int, default=8000, help="maximum number of epochs"
    )
    parser.add_argument("-dsfn", type=str, required=True, help="h5 dataset file")
    parser.add_argument(
        "-dsfolder", type=str, required=True, help="path dataset folder"
    )
    parser.add_argument(
        "-print",
        type=int,
        default=False,
        help="1: print to terminal; 0: redirect to file",
    )
    parser.add_argument(
        "-lrateg",
        type=float,
        default=1e-4,
        required=False,
        help="learning rate generator",
    )
    parser.add_argument(
        "-lrated", type=float, default=1e-4, help="learning rate discriminator"
    )
    parser.add_argument(
        "-hparams_file",
        type=str,
        default="0",
        help="Name of hparams file. '0' for default",
    )  # RSD: Figure this out.
    parser.add_argument(
        "-saveiter", type=int, default=250, help="save model every saveiter iterations"
    )

    return parser
