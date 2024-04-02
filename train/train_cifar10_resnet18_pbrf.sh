#!/bin/bash
HDIR=/h/nng/projects/OpenOOD
python3 ${HDIR}/main.py --config ${HDIR}/configs/datasets/cifar10/cifar10.yml \
    ${HDIR}/configs/networks/resnet18_32x32.yml \
    ${HDIR}/configs/preprocessors/base_preprocessor.yml \
    ${HDIR}/configs/pipelines/train/baseline.yml \
    ${HDIR}/configs/datasets/cifar10/cifar10_ood.yml \
    --seed 0 \
    --merge_option merge \
    --recorder.save_any_model False \
    --pbrf True \
    --dataset.train.batch_size 2048 \
    --optimizer.lr 1e-4 \
    --network.pretrained True \
    --network.checkpoint ${HDIR}/results/cifar10_resnet18_32x32_base_e100_lr0.1_default/s${1}/best.ckpt \
    --optimizer.num_epochs 100  ${@:2}
