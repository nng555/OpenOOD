import os
import numpy as np
import argparse

def get_order(name, temperature, ntrain):
    if 'cifar100' in name:
        dataset = 'cifar100'
    elif 'cifar10' in name:
        model = 'cifar10_resnet18_32x32_test_ood_ood_nak_default'
        dataset = 'cifar10'
    elif 'mnist' in name:
        model = 'mnist_lenet_test_ood_ood_nak_default'
        dataset = 'mnist'
    else:
        raise NotImplementedError

    labels = open(f'/h/nng/projects/OpenOOD/data/benchmark_imglist/{dataset}/train_{dataset}.txt').readlines()
    labels = np.array([int(l.strip().split(' ')[-1]) for i, l in enumerate(labels)])

    res = np.load(f'{name}_t{temperature}_best_raw.npz', allow_pickle=True)['extra'].item()
    logp = np.take_along_axis(-np.log(res['raw_probs']), labels[:, None], -1)[:, 0]
    tlogp = np.take_along_axis(-np.log(res['probs']), labels[:, None], -1)[:, 0]
    norm = (res['norm'] * res['probs']).sum(-1) / ntrain
    import ipdb; ipdb.set_trace()
    comp = logp / temperature + norm

    #prune_order = np.argsort(comp)
    #np.save(f'{dataset}_t{temperature}_best', prune_order)
    prune_order = np.argsort(np.take_along_axis(res['norm'], labels[:, None], -1)[:, 0])
    np.save(f'{dataset}_t{temperature}_best_self_if', prune_order)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', help='name of slurm folder')
    parser.add_argument('-t', help='temperature', type=int)
    parser.add_argument('-m', help='ntrain', type=int)
    args = parser.parse_args()
    get_order(args.n, args.t, args.m)
