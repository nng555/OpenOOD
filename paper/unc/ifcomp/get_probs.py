import os
import shutil
import argparse
import numpy as np
from scipy.special import softmax

corrs = ['brightness', 'cifar10', 'contrast', 'defocus_blur',
         'elastic_transform', 'fog', 'frost', 'gaussian_blur',
         'gaussian_noise', 'glass_blur', 'impulse_noise',
         'jpeg_compression', 'motion_blur', 'pixelate',
         'saturate', 'shot_noise', 'snow', 'spatter', 'speckle_noise',
         'zoom_blur']

def get_probs(name, alpha, temperature, prob_space):
    for c in corrs:
        res = np.load(os.path.join(name, c + '.npz'), allow_pickle=True)['extra'].item()

        if prob_space:
            probs = softmax(res['logits'] / temperature, -1)
            bnml = probs + probs * res['norm'] * alpha / 50000
            bnml /= bnml.sum(-1)[:, None]
        else:
            probs = softmax(res['logits'] / temperature, -1)
            logits = res['logits'] + res['norm'] * alpha / 50000
            bnml = softmax(logits / temperature, -1)

        np.save(c, {'probs': bnml})
        shutil.move(c + '.npy', c + '.npz')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', help='name of folder prefix')
    parser.add_argument('-a', help='norm weight', type=float)
    parser.add_argument('-t', help='temperature', type=float)
    parser.add_argument('-p', help='prob space normalization', action='store_true')
    args = parser.parse_args()
    get_probs(args.n, args.a, args.t, args.p)

