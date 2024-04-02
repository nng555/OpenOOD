#!/bin/bash
HDIR=/h/nng/projects/OpenOOD
python3 ${HDIR}/main.py --config ${HDIR}/configs/datasets/cifar10/cifar10_c.yml \
  ${HDIR}/configs/networks/resnet18_32x32.yml \
  ${HDIR}/configs/postprocessors/msp.yml \
  ${HDIR}/configs/pipelines/test/test_ood.yml \
  ${HDIR}/configs/datasets/cifar10/cifar10.yml \
  ${HDIR}/configs/preprocessors/base_preprocessor.yml \
  --network.pretrained True \
  --network.checkpoint ${HDIR}/paper/unc/models/${1}.ckpt ${@:2}

