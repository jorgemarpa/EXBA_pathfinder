import os, sys, glob
import argparse
import numpy as np
import pandas as pd
import warnings
from tqdm import tqdm
import matplotlib.pyplot as plt
import pickle

path = os.path.dirname(os.getcwd())

sys.path.append(path)
from exba import EXBA

parser = argparse.ArgumentParser(description="AutoEncoder")
parser.add_argument(
    "--quarter",
    dest="quarter",
    type=int,
    default=5,
    help="Which quarter.",
)
parser.add_argument(
    "--channel",
    dest="channel",
    type=int,
    default=1,
    help="List of files to be downloaded",
)
parser.add_argument(
    "--plot",
    dest="plot",
    action="store_true",
    default=False,
    help="Make plots.",
)
parser.add_argument(
    "--save",
    dest="save",
    action="store_true",
    default=True,
    help="Save models.",
)
parser.add_argument(
    "--dry-run",
    dest="dry_run",
    action="store_true",
    default=False,
    help="dry run.",
)
args = parser.parse_args()

def run_code(Q, CH):
    exba = EXBA(channel=CH, quarter=Q)
    exba.optimize_aperture(plot=args.plot)
    exba.find_aperture(space='pix-auto', cut=exba.optim_cut,
                       plot=args.plot, remove_blank=True)
    exba.contamination_metrics(plot=args.plot)
    exba.do_photometry()
    exba.apply_flatten()
    exba.apply_CBV(plot=False)
    exba.do_bls_search(n_boots=0, plot=False)
    if args.save:
        exba.store_data(all=True)

    del exba

    return

if __name__ == '__main__':
    print("Running EXBA tools for Q: %i Ch: %i" % (args.quarter, args.channel))
    if args.dry_run:
        print("Dry run mode, exiting...")
        sys.exit()
    run_code(Q=args.quarter, CH=args.channel)
