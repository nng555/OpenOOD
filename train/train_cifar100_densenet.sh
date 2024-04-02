#!/bin/bash
HDIR=/h/nng/projects/OpenOOD
python3 ${HDIR}/main.py --config ${HDIR}/configs/datasets/cifar100/cifar100.yml \
    ${HDIR}/configs/pipelines/train/baseline.yml \
    ${HDIR}/configs/networks/densenet.yml \
    ${HDIR}/configs/preprocessors/base_preprocessor.yml \
    --merge_option merge \
    --network.pretrained False \
    --optimizer.num_epochs 200  ${@:1}
