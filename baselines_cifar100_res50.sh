#!/bin/bash

HDIR=/h/nng/projects/OpenOOD
for baseline in msp mls odin knn ebo rmds; do 
  python3 ${HDIR}/main.py \
    --config ${HDIR}/configs/datasets/cifar100/cifar100_ood.yml \
    ${HDIR}/configs/networks/resnet50.yml \
    ${HDIR}/configs/postprocessors/${baseline}.yml \
    ${HDIR}/configs/pipelines/test/test_ood.yml \
    ${HDIR}/configs/datasets/cifar100/cifar100.yml \
    ${HDIR}/configs/preprocessors/base_preprocessor.yml \
    --network.pretrained True \
    --network.cifar True \
    --network.checkpoint ${HDIR}/results/checkpoints/cifar100_res50.ckpt ${@:2}
done

