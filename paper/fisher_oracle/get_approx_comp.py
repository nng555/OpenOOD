import os
from scipy.stats import pearsonr, spearmanr, kendalltau
import numpy as np
import json

#methods = ['acnml/step1', 'acnml/step5', 'nak/t1/d1e-12', 'nak/t1/d1e-3', 'nak/t1/d5e-4', 'nak/t1/d1e-4', 'nak/t1/d1e-5', 'nak/t1/d1e-6', 'nak/t1/d1e-7', 'nak/t1/d1e-8', 'nak/t1/d1e-9', 'nak/t1000', 'ntk', 'step/step1', 'step/step5']
methods = ['acnml/step1', 'acnml/step5', 'nak/t1/d1e-12', 'nak/t1000', 'nak/logit', 'ntk', 'step/step1', 'step/step5']
dsets = ['cifar10', 'cifar100', 'mnist']
#dsets = ['cifar10']

true_comp = json.load(open('true_comp.json'))

for m in methods:
    print(m)
    res = {}
    for dset in dsets:
        res[dset] = {}
        outs = np.load(os.path.join(m, dset + '.npz'), allow_pickle=True)['extra'].item()
        if 'regret' in outs:
            comp = outs['norm'] - outs['probs']
        elif m == 'nak/t1/d1e-12':
            comp = np.take_along_axis(outs['norm'], outs['probs'].argmax(-1)[:, None], -1)[:, None]
        else:
            comp = outs['norm'] * outs['probs']
        for k in true_comp[dset]:
            res[dset][k] = list(comp[int(k)])
    np.save(m + '_comp.json', res)

    total_true_comp = []
    total_approx_comp = []

    for dset in dsets:
        print(dset)
        dset_true_comp = []
        dset_approx_comp = []
        for k in true_comp[dset]:
            #min_c = np.argmin(true_comp[dset][k])
            tc = true_comp[dset][k].copy()
            #tc.pop(min_c)
            #res[dset][k].pop(min_c)
            #dset_true_comp.extend(tc)
            #dset_approx_comp.extend(res[dset][k])
            dset_true_comp.append(np.sum(tc))
            dset_approx_comp.append(np.sum(res[dset][k]))
        print(pearsonr(dset_true_comp, dset_approx_comp))
        total_true_comp.extend(dset_true_comp)
        total_approx_comp.extend(dset_approx_comp)
        #if m == 'nak/t1' or m == 'ntk':
        #    import ipdb; ipdb.set_trace()

    print("all")
    print(pearsonr(total_true_comp, total_approx_comp))
    print()
