#!/bin/bash
HDIR=/h/nng/projects/OpenOOD
python3 ${HDIR}/main.py --config ${HDIR}/configs/datasets/cifar10/cifar10.yml \
    ${HDIR}/configs/networks/mlp.yml \
    ${HDIR}/configs/preprocessors/base_preprocessor.yml \
    ${HDIR}/configs/pipelines/train/baseline.yml \
    --merge_option merge \
    --optimizer.lr 0.001 \
    --optimizer.num_epochs 200 \
    --network.im_size 3072 \
    --seed 0 
