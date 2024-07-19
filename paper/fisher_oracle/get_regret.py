import os
import json
import glob
import numpy as np
from collections import defaultdict

#datasets = ['cifar10', 'cifar100', 'mnist']
datasets = ['cifar10']

res = {}

for dset in datasets:
    res[dset] = {}
    for f in glob.glob(f'orig_out/*{dset},*'):

        comp_index = int(f.split('comp_index_')[-1].split(',')[0])
        comp_label = int(f.split('comp_label_')[-1].split(',')[0])

        out_logs = []
        for _out_log in glob.glob(os.path.join(f, 'log', '*.out')):
            out_logs.append(int(_out_log.split('/')[-1].split('.out')[0]))
        if len(out_logs) == 0:
            print(f"No logs found for {f}")
            continue

        out_log = os.path.join(f, 'log', str(max(out_logs)) + '.out')

        regret = None
        for line in open(out_log).readlines():
            if 'Regret' in line:
                regret = float(line.strip().split('Regret: ')[-1])
        if regret is None:
            print(f"No regret found for {f}")
            continue

        if comp_index not in res[dset]:
            res[dset][comp_index] = [0 for _ in range(10)]

        if res[dset][comp_index][comp_label] == 0:
            res[dset][comp_index][comp_label] = regret

with open('true_regret.json', 'w') as of:
    json.dump(res, of)
