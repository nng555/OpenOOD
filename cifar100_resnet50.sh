#!/bin/bash
HDIR=/h/nng/projects/OpenOOD
python3 ${HDIR}/main.py \
  --config ${HDIR}/configs/datasets/cifar100/cifar100_ood.yml \
  ${HDIR}/configs/networks/resnet50.yml \
  ${HDIR}/configs/postprocessors/nak.yml \
  ${HDIR}/configs/pipelines/test/test_ood.yml \
  ${HDIR}/configs/datasets/cifar100/cifar100.yml \
  ${HDIR}/configs/preprocessors/base_preprocessor.yml \
  --network.cifar True \
  --network.pretrained True \
  --network.checkpoint /fs01/home/nng/slurm/2023-10-18/run_bash/cifar100,train/results/cifar100_resnet50_base_e51_lr0.004_default/s0/best.ckpt ${@:1}
