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
  --postprocessor.postprocessor_args.jac_chunk_size 1 \
  --postprocessor.postprocessor_args.class_chunk_size 50 \
  --network.checkpoint /h/nng/projects/OpenOOD/results/checkpoints/cifar100_res50.ckpt ${@:1}
