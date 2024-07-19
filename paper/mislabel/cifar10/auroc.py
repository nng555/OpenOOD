# coding: utf-8
from sklearn.metrics import f1_score, roc_auc_score, precision_recall_curve
import numpy as np
import argparse


def analyze(noise, last):
    if last:
        time = 'last'
    else:
        time = 'best'

    #mt1_res = np.load(f'{noise}/{time}_mis_t1.npz', allow_pickle=True)['extra'].item()
    #mt1000_res = np.load(f'{noise}/{time}_mis_t1000.npz', allow_pickle=True)['extra'].item()
    t1_res = np.load(f'{noise}/{time}_t1.npz', allow_pickle=True)['extra'].item()
    t1000_res = np.load(f'{noise}/{time}_t1000.npz', allow_pickle=True)['extra'].item()

    el2n_res = np.load(f'{noise}/el2n.npy')
    tracin_res = np.load(f'{noise}/tracin.npy')
    grand_res = np.load(f'{noise}/grand.npy')

    cnoise = np.load(f'/h/nng/projects/OpenOOD/data/benchmark_imglist/cifar10/train_cifar10_{noise}.npy', allow_pickle=True).item()

    cmap = cnoise['cmap']
    labels = open('/h/nng/projects/OpenOOD/data/benchmark_imglist/cifar10/train_cifar10.txt').readlines()
    clabels = np.array([int(cmap.get(l, l).strip().split(' ')[-1]) for l in labels])

    cidxs = cnoise['cidxs']
    ckeys = {c[0] for c in cidxs}

    rlabels = [i in ckeys for i in range(50000)]
    t1_comp = (t1_res['probs'] * t1_res['norm']).sum(-1)
    t1_self_if = np.take_along_axis(t1_res['norm'], clabels[:, None], -1)[:, 0]
    t1000_comp = (t1000_res['probs'] * t1000_res['norm']).sum(-1)
    t1000_self_if = np.take_along_axis(t1000_res['norm'], clabels[:, None], -1)[:, 0]

    t1_logp = np.take_along_axis(-np.log(t1_res['probs']), clabels[:, None], -1)[:, 0]
    t1000_logp = np.take_along_axis(-np.log(t1000_res['probs']), clabels[:, None], -1)[:, 0]

    print("self_if")
    print(roc_auc_score(rlabels, t1_self_if))
    print(roc_auc_score(rlabels, t1000_self_if))
    print(roc_auc_score(rlabels, t1_comp))
    print(roc_auc_score(rlabels, t1000_comp))
    print(roc_auc_score(rlabels, t1_logp))
    print(roc_auc_score(rlabels, t1000_logp))
    print("if-comp")
    print(roc_auc_score(rlabels, t1_comp/50000))
    print(roc_auc_score(rlabels, t1000_comp/50000))
    print(roc_auc_score(rlabels, t1_logp + t1_comp/50000))
    print(roc_auc_score(rlabels, t1_logp + t1000_comp/50000))
    print(roc_auc_score(rlabels, t1000_logp + t1_comp/50000))
    print(roc_auc_score(rlabels, t1000_logp + t1000_comp/50000))
    print('el2n')
    print(roc_auc_score(rlabels, el2n_res))
    print('grand')
    print(roc_auc_score(rlabels, grand_res))
    print('tracin')
    print(roc_auc_score(rlabels, tracin_res))

    tracin_rank = np.array([i in ckeys for i in tracin_res.argsort()])
    grand_rank = np.array([i in ckeys for i in grand_res.argsort()])
    el2n_rank = np.array([i in ckeys for i in el2n_res.argsort()])
    comp_rank = np.array([i in ckeys for i in (t1_logp/1000 + t1000_comp/50000).argsort()])
    if_rank = np.array([i in ckeys for i in t1_self_if.argsort()])

    import ipdb; ipdb.set_trace()

def get_f1(rlabels, scores):
    precision, recall, thresholds = precision_recall_curve(rlabels, scores)
    f1_scores = 2 * recall * precision / (precision + recall + 1e-9)
    return np.max(f1_scores)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', help='noise')
    parser.add_argument('-l', help='last', action='store_true')
    args = parser.parse_args()

    analyze(args.n, args.l)
