import numpy as np
import torch
import os
import matplotlib.pyplot as plt
import argparse

import datashader as ds
import pandas as pd
import colorcet as cc

parser = argparse.ArgumentParser()
parser.add_argument('-o', '--dir', help='directory of npz to process')
parser.add_argument('-i', '--input', help='spectrum file')
parser.add_argument('-d', '--damping', help='damping amount', type=float)
#parser.add_argument('-f', '--filter', help='filter eigenvalues below threshold', type=float, default=0.0)
#parser.add_argument('-p', '--portion', help='fraction of top eigenvalues to show', type=float, default=1.0)
args = parser.parse_args()

spec_dir = '/h/nng/projects/OpenOOD/results/ekfac'

spec = torch.load(os.path.join(spec_dir, args.input), map_location='cpu')
spec = torch.cat(spec).numpy()

for f in os.listdir(args.dir):
    if '.npz' not in f:
        continue
    feats = np.load(os.path.join(args.dir, f), allow_pickle=True)['extra'].item()
    if 'eigenfeat' not in feats:
        continue
    feats = feats['eigenfeat']
    mask = np.logical_or(feats == 0, spec == 0)
    mask = np.logical_or(mask, spec < args.damping)

    #df = pd.DataFrame({'x': spec[~mask], 'y': feats[~mask]})
    #cvs = ds.Canvas(plot_width=700, plot_height=700, x_axis_type='log', y_axis_type='log')  # auto range or provide the `bounds` argument
    #agg = cvs.points(df, 'x', 'y')  # this is the histogram
    #img = ds.tf.set_background(ds.tf.shade(agg, how="log", cmap=cc.fire), "black").to_pil()  # create a rasterized image
    #plt.imshow(img)
    plt.clf()
    plt.cla()
    plt.scatter(1. / (spec[~mask] + args.damping), feats[~mask], s=2, alpha=0.2)
    plt.xscale('log')
    plt.yscale('log')
    plt.savefig(os.path.join(args.dir, f.split('.')[0] + '.png'))

