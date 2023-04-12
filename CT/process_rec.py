import argparse
from data_making import *

"""
Process EQNR .raw-file reconstruction to h5 deep learning dataset
"""

parser = argparse.ArgumentParser(description="EQNR processing of .raw reconstruction")

parser.add_argument("-root", type=str, required=True, help="root path")
parser.add_argument(
    "-oroot", type=str, default="", required=False, help="output root path"
)
parser.add_argument("-rawname", type=str, required=True, help="Input name")
parser.add_argument("-expname", type=str, required=True, help="Experiment name")
parser.add_argument("-vis", type=bool, default=True, help="Visual inspection")

args, unparsed = parser.parse_known_args()


output = ReconstructionsDataCT(args.oroot, args.expname)

raw_instance = EquinorReconstructions(args.root, args.rawname, args.oroot, args.expname)

output.add_item(raw_instance)


if args.vis:
    output.visualise()
