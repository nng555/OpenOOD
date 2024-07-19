import os
import pickle
import argparse
import numpy as np

CIFAR10_MAP= {
    1: 9,
    9: 1,
    2: 0,
    0: 2,
    3: 5,
    5: 3,
    7: 4,
    4: 7,
    6: 8,
    8: 6,
}

CIFAR100_MAP = pickle.load(open('asym_transform.pkl', 'rb'))

def generate_mislabel(filename, nclasses, fprob, symmetric):
    files = open(filename).readlines()
    nflip = int(fprob * len(files))

    fidxs = np.random.choice(len(files), nflip, replace=False)
    orig_labels = [int(files[fidx].strip().split(' ')[-1]) for fidx in fidxs]

    if symmetric:
        flips = np.random.choice(nclasses - 1, len(files), replace=True)
        new_labels = [int((ol + flip + 1) % nclasses) for ol, flip in zip(orig_labels, flips)]
    else:
        if nclasses == 10:
            new_labels = [CIFAR10_MAP[ol] for ol in orig_labels]
        elif nclasses == 100:
            new_labels = [CIFAR100_MAP[ol] for ol in orig_labels]
        else:
            raise NotImplementedError

    fname = filename.split('.txt')[0]
    if symmetric:
        fname += '_sym'
    else:
        fname += '_asym'
    fname += f'_{fprob}'

    cidxs = np.stack([fidxs, orig_labels, new_labels]).T
    cmap = {files[fidx]: files[fidx].split(' ')[0] + ' ' + str(clabel) + '\n' for fidx, clabel in zip(fidxs, new_labels)}
    res = {
        'cidxs': cidxs,
        'cmap': cmap,
    }
    np.save(fname, res)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', help='imglist filename')
    parser.add_argument('-n', type=int, help='nclasses')
    parser.add_argument('-s', help='symmetric', action='store_true')
    parser.add_argument('-p', type=float, help='probability of flip')
    args = parser.parse_args()
    generate_mislabel(args.f, args.n, args.p, args.s)
