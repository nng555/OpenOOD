#!/bin/bash
HDIR=/h/nng/projects/OpenOOD
python3 ${HDIR}/main.py --config ${HDIR}/configs/datasets/mnist/mnist.yml \
    ${HDIR}/configs/networks/lenet.yml \
    ${HDIR}/configs/preprocessors/base_preprocessor.yml \
    ${HDIR}/configs/pipelines/train/baseline.yml \
    --merge_option merge \
    --optimizer.lr 0.001 \
    --optimizer.num_epochs 200 \
    --seed 0 ${@:1}
