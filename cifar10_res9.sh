#!/bin/bash
HDIR=/h/nng/projects/OpenOOD
python3 ${HDIR}/main.py --config ${HDIR}/configs/datasets/cifar10/cifar10_ood.yml \
  ${HDIR}/configs/networks/resnet9_32x32.yml \
  ${HDIR}/configs/postprocessors/nak.yml \
  ${HDIR}/configs/pipelines/test/test_ood.yml \
  ${HDIR}/configs/datasets/cifar10/cifar10.yml \
  ${HDIR}/configs/preprocessors/base_preprocessor.yml \
  --network.pretrained True \
  --network.checkpoint ${HDIR}/models/checkpoints/cifar10_res9_s${1}_noaug.ckpt ${@:2}
