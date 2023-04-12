import argparse
from preprocessing import *

"""
Do preprocessing of a EQNR scan
"""

parser = argparse.ArgumentParser(description="EQNR preprocessing of obtained CT scan")

parser.add_argument("-root", type=str, required=True, help="root path")
parser.add_argument(
    "-oroot", type=str, default="", required=False, help="output root path"
)
parser.add_argument("-expname", type=str, required=True, help="Experiment name")
parser.add_argument(
    "-correction", type=str, default="", required=False, help="Path corrections"
)
parser.add_argument("-numProj", type=int, default=0, help="number of projections")
parser.add_argument(
    "-dynamic", type=bool, required=False, default=True, help="Dynamic or static scan"
)
parser.add_argument(
    "-geometry",
    type=str,
    required=False,
    default="",
    help="Geometry path if not default",
)
parser.add_argument(
    "-roi",
    type=int,
    required=False,
    default=0,
    nargs="+",
    help="ROI path if not default",
)
parser.add_argument(
    "-rot", type=float, required=False, default=0, help="Rotation of projections"
)
parser.add_argument("-vis", type=bool, default=True, help="Visual inspection")
# parser.add_argument("-step0", type=str, required=True, help="Step 0 nsiprg filename") #Untrue


args, unparsed = parser.parse_known_args()


if args.correction == "":
    args.correction = None

if args.geometry == "":
    args.geometry = None

if args.roi == 0:
    args.roi = None

if args.dynamic:
    print("Dynamic scan")
    preprocess = DynamicProjectionsEQNR(
        args.root,
        args.expname,
        args.oroot,
        args.numProj,
        args.correction,
        geometry=args.geometry,
        roi=args.roi,
        rotation=args.rot,
    )

else:
    print("Static scan")
    preprocess = ProjectionsEQNR(
        args.root,
        args.expname,
        args.oroot,
        args.numProj,
        args.correction,
        geometry=args.geometry,
        roi=args.roi,
        rotation=args.rot,
    )

preprocess()

if args.vis:
    preprocess.visualise()
