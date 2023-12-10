#!/bin/bash

HDIR=/h/nng/projects/OpenOOD
for baseline in msp mls odin knn ebo; do 
  python3 ${HDIR}/main.py \
    --config ${HDIR}/configs/datasets/cifar10/cifar10_ood.yml \
    ${HDIR}/configs/networks/vgg.yml \
    ${HDIR}/configs/postprocessors/${baseline}.yml \
    ${HDIR}/configs/pipelines/test/test_ood.yml \
    ${HDIR}/configs/datasets/cifar10/cifar10.yml \
    ${HDIR}/configs/preprocessors/base_preprocessor.yml \
    --network.pretrained True \
    --network.checkpoint /fs01/home/nng/slurm/2023-11-10/run_bash/bn,cifar10,lr0.05,train,vgg/results/cifar10_vgg_base_e200_lr0.05_default/s0/best.ckpt \
    --network.use_bn True
done
