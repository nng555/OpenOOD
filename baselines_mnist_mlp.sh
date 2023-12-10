#!/bin/bash

HDIR=/h/nng/projects/OpenOOD
for baseline in msp mls odin knn ebo; do 
  python3 ${HDIR}/main.py \
    --config ${HDIR}/configs/datasets/mnist/mnist_ood.yml \
    ${HDIR}/configs/networks/mlp.yml \
    ${HDIR}/configs/postprocessors/${baseline}.yml \
    ${HDIR}/configs/pipelines/test/test_ood.yml \
    ${HDIR}/configs/datasets/mnist/mnist.yml \
    ${HDIR}/configs/preprocessors/base_preprocessor.yml \
    --network.pretrained True \
    --network.checkpoint /fs01/home/nng/projects/OpenOOD/results/checkpoints/mnist_mlp.ckpt
done
