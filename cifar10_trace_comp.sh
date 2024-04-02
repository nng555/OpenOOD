#!/bin/bash
HDIR=/h/nng/projects/OpenOOD
python3 ${HDIR}/main.py --config ${HDIR}/configs/datasets/cifar10/cifar10_ood.yml \
  ${HDIR}/configs/networks/resnet18_32x32.yml \
  ${HDIR}/configs/postprocessors/nak.yml \
  ${HDIR}/configs/pipelines/test/test_ood.yml \
  ${HDIR}/configs/datasets/cifar10/cifar10.yml \
  ${HDIR}/configs/preprocessors/base_preprocessor.yml \
  --network.pretrained True \
  --postprocessor.postprocessor_args.damping 1e-12 \
  --network.checkpoint /h/nng/slurm/2024-02-01/run_bash/${1}/results/cifar10_resnet18_32x32_base_e200_lr0.1_default/s0/best.ckpt ${@:2}
