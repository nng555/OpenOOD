#!/bin/bash

HDIR=/h/nng/projects/OpenOOD
for baseline in msp mls odin knn ebo; do 
  python3 ${HDIR}/main.py \
    --config ${HDIR}/configs/datasets/cifar10/cifar10_ood.yml \
    ${HDIR}/configs/networks/nin.yml \
    ${HDIR}/configs/postprocessors/${baseline}.yml \
    ${HDIR}/configs/pipelines/test/test_ood.yml \
    ${HDIR}/configs/datasets/cifar10/cifar10.yml \
    ${HDIR}/configs/preprocessors/base_preprocessor.yml \
    --network.pretrained True \
    --network.checkpoint /fs01/home/nng/slurm/2023-11-08/run_bash/cifar10,nin,train/results/cifar10_nin_base_e200_lr0.1_default/s0/best.ckpt
done
