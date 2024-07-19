import os
#import torch
#from torchmetrics.classification import MulticlassCalibrationError
import glob
import argparse
import numpy as np

def get_reliability(name, corr, sev):
    labels = np.load('/h/nng/projects/OpenOOD/data/images_classic/CIFAR-10-C/labels.npy')
    c10_labels = open('/h/nng/projects/OpenOOD/data/benchmark_imglist/cifar10/test_cifar10.txt').readlines()
    c10_labels = np.array([int(f.strip().split(' ')[-1]) for f in c10_labels])

    res = np.load(os.path.join(name, corr + '.npz'), allow_pickle=True)
    if 'extra' in res:
        res = res['extra']
    res = res.item()
    if name == 'acnml':
        res['probs'] = res['norm'] / res['norm'].sum(-1)[:, None]

    if corr == 'cifar10':
        c_probs = res['probs']
        c_corr = (c10_labels == c_probs.argmax(-1))
    else:
        ssev = sev * 10000
        esev = (sev + 1) * 10000

        c_probs = res['probs'][ssev:esev]
        c_corr = (labels[ssev:esev] == c_probs.argmax(-1))

    sort_order = c_probs.max(-1).argsort()
    c_probs = c_probs[sort_order]
    c_corr = c_corr[sort_order]

    bin_size = int(np.ceil(len(c_probs) / 20))

    confs = []
    accs = []
    for j in range(20):
        sbin = j * bin_size
        ebin = (j + 1) * bin_size
        confs.append(c_probs[sbin:ebin].max(-1).mean())
        accs.append(c_corr[sbin:ebin].mean())

    for j in range(20):
        print(f"{confs[j]}, {accs[j]}")

    print(np.mean([np.abs(c - a) for c, a in zip(confs, accs)]))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', help='name of method')
    parser.add_argument('-c', help='corruption name')
    parser.add_argument('-s', help='severity level', type=int)
    args = parser.parse_args()
    get_reliability(args.n, args.c, args.s)
