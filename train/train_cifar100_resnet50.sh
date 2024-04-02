#!/bin/bash
HDIR=/h/nng/projects/OpenOOD
python3 ${HDIR}/main.py --config ${HDIR}/configs/datasets/cifar100/cifar100.yml \
    ${HDIR}/configs/pipelines/train/baseline.yml \
    ${HDIR}/configs/networks/resnet50.yml \
    ${HDIR}/configs/preprocessors/base_preprocessor.yml \
    --seed 10 \
    --merge_option merge \
    --preprocessor.augment True \
    --network.pretrained False \
    --optimizer.num_epochs 200 \
    --network.cifar True  ${@:1}
