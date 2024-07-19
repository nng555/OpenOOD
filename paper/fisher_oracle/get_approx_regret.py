import os
from scipy.stats import pearsonr, spearmanr, kendalltau
import numpy as np
import json

methods = ['acnml/step1', 'acnml/step5', 'nak/t1', 'nak/t1000', 'ntk', 'step/step1', 'step/step5']
#dsets = ['cifar10', 'cifar100', 'mnist']
dsets = ['cifar10']

true_regret = json.load(open('true_regret.json'))

for m in methods:
    print(m)
    res = {}
    for dset in dsets:
        res[dset] = {}
        outs = np.load(os.path.join(m, dset + '.npz'), allow_pickle=True)['extra'].item()
        if 'regret' in outs:
            regret = outs['regret']
        else:
            regret = outs['norm']
        for k in true_regret[dset]:
            res[dset][k] = list(regret[int(k)])
    np.save(m + '_regret.json', res)

    total_true_regret = []
    total_approx_regret = []

    for dset in dsets:
        print(dset)
        dset_true_regret = []
        dset_approx_regret = []
        for k in true_regret[dset]:
            #min_r = np.argmin(true_regret[dset][k])
            tr = true_regret[dset][k].copy()
            #tr.pop(min_r)
            #res[dset][k].pop(min_r)
            dset_true_regret.extend(tr)
            dset_approx_regret.extend(res[dset][k])
        print(spearmanr(dset_true_regret, dset_approx_regret))
        total_true_regret.extend(dset_true_regret)
        total_approx_regret.extend(dset_approx_regret)

    print("all")
    print(spearmanr(total_true_regret, total_approx_regret))
    print()
