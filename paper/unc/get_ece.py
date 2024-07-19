import os
#import torch
#from torchmetrics.classification import MulticlassCalibrationError
import glob
import argparse
import numpy as np

corrs = ['brightness', 'cifar10', 'contrast', 'defocus_blur',
         'elastic_transform', 'fog', 'frost', 'gaussian_blur',
         'gaussian_noise', 'glass_blur', 'impulse_noise',
         'jpeg_compression', 'motion_blur', 'pixelate',
         'saturate', 'shot_noise', 'snow', 'spatter', 'speckle_noise',
         'zoom_blur']
corrs = np.array(corrs)

def get_ece(name, offset):
    labels = np.load('/h/nng/projects/OpenOOD/data/images_classic/CIFAR-10-C/labels.npy')
    c10_labels = open('/h/nng/projects/OpenOOD/data/benchmark_imglist/cifar10/test_cifar10.txt').readlines()
    c10_labels = np.array([int(f.strip().split(' ')[-1]) for f in c10_labels])

    #ece = MulticlassCalibrationError(10, n_bins=20)

    outs = [[] for _ in range(5)]
    accs = [[] for _ in range(5)]

    for c in corrs:
        res = np.load(os.path.join(name, c + '.npz'), allow_pickle=True)
        if 'extra' in res:
            res = res['extra']
        res = res.item()
        if name == 'acnml':
            res['probs'] = res['norm'] / res['norm'].sum(-1)[:, None]

        #print(c)
        if c == 'cifar10':
            c_probs = res['probs']
            c_corr = c10_labels == c_probs.argmax(-1)

            sort_order = c_probs.max(-1).argsort()
            c_probs = c_probs[sort_order]
            c_corr = c_corr[sort_order]

            ece = 0

            bin_size = int(9000 / 20)

            for j in range(20):
                sbin = j * bin_size
                ebin = (j + 1) * bin_size
                confidence = c_probs[sbin:ebin].max(-1).mean()
                accuracy = c_corr[sbin:ebin].mean()
                ece += np.abs(confidence - accuracy) / 20

            print(ece)
            print(c_corr.mean())
            #print(ece(torch.Tensor(res['probs']), torch.Tensor(c10_labels)).item())
        else:
            for i in range(5):
                ssev = i * 10000
                esev = (i + 1) * 10000

                c_probs = res['probs'][ssev:esev]
                c_corr = (labels[ssev:esev] == c_probs.argmax(-1))
                accs[i].append(c_corr.mean())

                sort_order = c_probs.max(-1).argsort()
                c_probs = c_probs[sort_order]
                c_corr = c_corr[sort_order]

                ece = 0

                bin_size = int(10000 / 20)

                for j in range(20):
                    sbin = j * bin_size
                    ebin = (j + 1) * bin_size
                    confidence = c_probs[sbin:ebin].max(-1).mean()
                    accuracy = c_corr[sbin:ebin].mean()
                    ece += np.abs(confidence - accuracy) / 20

                outs[i].append(ece)
                #outs[i].append(ece(torch.Tensor(res['probs'][ssev:esev]), torch.Tensor(labels[ssev:esev])).item())

    outs = np.array(outs)
    for i, o in enumerate(outs):
        fq = np.quantile(o, [0.25, 0.75])
        print(f"{i + 1 + offset}, {np.median(o)}, {np.median(o) - fq[0]}, {fq[1] - np.median(o)}")
        #print(f"{i + 1 + offset}, {np.median(o)}, {np.median(o) - o[4]}, {o[14] - np.median(o)}")
        #probs = res['probs'].reshape(5, 10000, 10)

    print()
    for i, a in enumerate(accs):
        fq = np.quantile(a, [0.25, 0.75])
        #print(f"{i + 1 + offset}, {np.median(a)}, {np.median(a) - fq[0]}, {fq[1] - np.median(a)}")
        print(f"{i + 1 + offset}, {np.mean(a)}, {np.std(a)}, {np.std(a)}")
        #print(f"{np.mean(a)}")

    #for i, o in enumerate(outs):
        #print(np.sort(o))
        #print(corrs[o.argsort()])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', help='name of method')
    parser.add_argument('-o', help='offset', type=float)
    args = parser.parse_args()
    get_ece(args.n, args.o)
