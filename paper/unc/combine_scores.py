import os
import glob
import argparse
import numpy as np

corrs = ['brightness', 'cifar10', 'contrast', 'defocus_blur',
         'elastic_transform', 'fog', 'frost', 'gaussian_blur',
         'gaussian_noise', 'glass_blur', 'impulse_noise',
         'jpeg_compression', 'motion_blur', 'pixelate',
         'saturate', 'shot_noise', 'snow', 'spatter', 'speckle_noise',
         'zoom_blur']

def combine_scores(prefix):
    labels = np.load('/h/nng/projects/OpenOOD/data/images_classic/CIFAR-10-C/labels.npy')
    c10_labels = open('/h/nng/projects/OpenOOD/data/benchmark_imglist/cifar10/test_cifar10.txt').readlines()
    c10_labels = np.array([int(f.strip().split(' ')[-1]) for f in c10_labels])
    for c in corrs:
        out = None
        for i, fname in enumerate(glob.glob(prefix + '*')):
            res = np.load(os.path.join(fname, c + '.npz'), allow_pickle=True)['extra'].item()
            if out is None:
                out = res['probs']
            else:
                out += res['probs']
        out /= (i + 1)
        if c == 'cifar10':
            correct = (out.argmax(-1) == c10_labels)
        else:
            correct = (out.argmax(-1) == labels)
        np.save(os.path.join('/'.join(prefix.split('/')[:-1]), c), {'probs': out, 'correct': correct})
        print(c)
        print(correct.mean())

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', help='name of folder prefix')
    args = parser.parse_args()
    combine_scores(args.n)
