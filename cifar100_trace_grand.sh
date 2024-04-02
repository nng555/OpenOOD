#!/bin/bash
HDIR=/h/nng/projects/OpenOOD
python3 ${HDIR}/main.py --config ${HDIR}/configs/datasets/cifar10/cifar10_ood.yml \
  ${HDIR}/configs/networks/resnet18_32x32.yml \
  ${HDIR}/configs/postprocessors/nak.yml \
  ${HDIR}/configs/pipelines/test/test_ood.yml \
  ${HDIR}/configs/datasets/cifar100/cifar100.yml \
  ${HDIR}/configs/preprocessors/base_preprocessor.yml \
  --network.pretrained True \
  --postprocessor.postprocessor_args.natural False \
  --postprocessor.postprocessor_args.all_classes False \
  --network.checkpoint /h/nng/slurm/2024-01-25/run_bash/${1}/last.ckpt ${@:2}
