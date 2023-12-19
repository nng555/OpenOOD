#!/bin/bash

HDIR=/h/nng/projects/OpenOOD
for baseline in msp mls odin knn ebo rmds vim; do 
  python3 ${HDIR}/main.py \
    --config ${HDIR}/configs/datasets/cifar10/cifar10_ood.yml \
    ${HDIR}/configs/networks/bit.yml \
    ${HDIR}/configs/postprocessors/${baseline}.yml \
    ${HDIR}/configs/pipelines/test/test_ood.yml \
    ${HDIR}/configs/datasets/cifar10/cifar10.yml \
    ${HDIR}/configs/preprocessors/base_preprocessor.yml \
    --network.pretrained True \
    --network.model BiT-M-R50x1 \
    --dataset.pre_size 128 \
    --dataset.image_size 128 \
    --network.checkpoint ${HDIR}/results/checkpoints/cifar10_bit.cpt ${@:1}
done
