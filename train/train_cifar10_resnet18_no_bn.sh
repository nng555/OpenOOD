#!/bin/bash
HDIR=/h/nng/projects/OpenOOD
python3 ${HDIR}/main.py --config ${HDIR}/configs/datasets/cifar10/cifar10.yml \
    ${HDIR}/configs/networks/resnet18_32x32.yml \
    ${HDIR}/configs/preprocessors/base_preprocessor.yml \
    ${HDIR}/configs/pipelines/train/baseline.yml \
    --merge_option merge \
    --optimizer.num_epochs 200 \
    --seed 0 \
    --network.use_bn False
