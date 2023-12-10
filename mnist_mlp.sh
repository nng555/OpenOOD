#!/bin/bash
HDIR=/h/nng/projects/OpenOOD
python3 ${HDIR}/main.py \
  --config ${HDIR}/configs/datasets/mnist/mnist_ood.yml \
  ${HDIR}/configs/networks/mlp.yml \
  ${HDIR}/configs/postprocessors/nak.yml \
  ${HDIR}/configs/pipelines/test/test_ood.yml \
  ${HDIR}/configs/datasets/mnist/mnist.yml \
  ${HDIR}/configs/preprocessors/base_preprocessor.yml \
  --network.pretrained True \
  --network.checkpoint ${HDIR}/results/checkpoints/mnist_mlp.ckpt $*
