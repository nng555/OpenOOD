#!/bin/bash
HDIR=/h/nng/projects/OpenOOD
python3 ${HDIR}/main.py --config ${HDIR}/configs/datasets/cifar10/cifar10_ood.yml \
  ${HDIR}/configs/networks/resnet18_32x32.yml \
  ${HDIR}/configs/postprocessors/nak.yml \
  ${HDIR}/configs/pipelines/test/test_ood.yml \
  ${HDIR}/configs/datasets/cifar10/cifar10.yml \
  ${HDIR}/configs/preprocessors/base_preprocessor.yml \
  --network.pretrained True \
  --network.checkpoint /fs01/home/nng/slurm/2023-10-18/run_bash/cifar10,train/results/cifar10_resnet18_32x32_base_e200_lr0.1_default/s0/best_epoch144_acc0.9260.ckpt \
  --network.use_bn False ${@:1}
