import glob, os
import shutil
import argparse
import wget

import numpy as np
from astropy.io import fits

main_path = os.path.dirname(os.getcwd())

parser = argparse.ArgumentParser(description='AutoEncoder')
parser.add_argument('--dry-run', dest='dry_run', action='store_true',
                    default=False,
                    help='Dry run')
parser.add_argument('--file-list', dest='file_list', type=str,
                    help='List of files to be downloaded')
args = parser.parse_args()

def download_exba_files(file_list):
    # creates a temp file
    if not os.path.isdir('%s/data/temp' % main_path):
        os.mkdir('%s/data/temp' % main_path)

    urls = np.loadtxt(file_list, dtype=str)
    # check file isnt empty
    print(len(urls))

    for i, url in enumerate(urls[:100]):
        print(url)
        fname = url.split('/')[-1]
        out = '%s/data/temp/%s' % (main_path, fname)
        if os.path.isfile(out):
            continue
        else:
            wget.download(url, out=out)

    return

def create_tree_file():
    fnames = glob.glob('%s/data/temp/*.fits.gz' % main_path)

    # load quarters and channels
    quarters, channels = [], []
    for i, f in enumerate(fnames):
        temp = fits.open(f)
        quarters.append(temp[0].header["QUARTER"])
        channels.append(temp[0].header["CHANNEL"])

    # create directory tree
    for ch in set(channels):
        if not os.path.isdir('%s/data/EXBA/%s' % (main_path, str(ch))):
            os.mkdir('%s/data/EXBA/%s' % (main_path, str(ch)))
        for q in set(quarters):
            if not os.path.isdir('%s/data/EXBA/%s/%s' % (main_path, str(ch), str(q))):
                os.mkdir('%s/data/EXBA/%s/%s' % (main_path, str(ch), str(q)))

    for f, q, ch in zip(fnames, quarters, channels):
        name = f.split('/')[-1]
        out = '%s/data/EXBA/%s/%s/%s' % (main_path, str(ch), str(q), name)
        shutil.move(f, out)

if __name__ == '__main__':
    print("Downloading files from provided list off URLs")
    download_exba_files(args.file_list)
    print("Done!")

    print("Creating tree directory")
    create_tree_file()
    print("Done!")
