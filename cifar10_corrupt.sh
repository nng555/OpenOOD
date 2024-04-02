#!/bin/bash
HDIR=/h/nng/projects/OpenOOD
python3 ${HDIR}/main.py --config ${HDIR}/configs/datasets/cifar10/cifar10_ood.yml \
  ${HDIR}/configs/networks/resnet18_32x32.yml \
  ${HDIR}/configs/postprocessors/nak.yml \
  ${HDIR}/configs/pipelines/test/test_ood.yml \
  ${HDIR}/configs/datasets/cifar10/cifar10.yml \
  ${HDIR}/configs/preprocessors/base_preprocessor.yml \
  --dataset.corruption_path ${1} \
  --postprocessor.postprocessor_args.state_path /h/nng/projects/OpenOOD/results/ekfac/cifar10_${1}_${2}_\${temp} \
  --network.pretrained True \
  --network.checkpoint ${HDIR}/models/checkpoints/cifar10_${1}_${2}.ckpt ${@:3}
