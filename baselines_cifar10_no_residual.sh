#!/bin/bash

HDIR=/h/nng/projects/OpenOOD
for baseline in msp mls odin knn ebo; do 
  python3 ${HDIR}/main.py \
    --config ${HDIR}/configs/datasets/cifar10/cifar10_ood.yml \
    ${HDIR}/configs/networks/resnet18_32x32.yml \
    ${HDIR}/configs/postprocessors/${baseline}.yml \
    ${HDIR}/configs/pipelines/test/test_ood.yml \
    ${HDIR}/configs/datasets/cifar10/cifar10.yml \
    ${HDIR}/configs/preprocessors/base_preprocessor.yml \
    --network.pretrained True \
    --network.checkpoint /fs01/home/nng/slurm/2023-11-07/run_bash/bn,cifar10,no_residual,train/results/cifar10_resnet18_32x32_base_e200_lr0.1_default/s0/best_epoch130_acc0.9500.ckpt \
    --network.residual False
done
