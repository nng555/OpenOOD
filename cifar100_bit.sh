#!/bin/bash
HDIR=/h/nng/projects/OpenOOD
python3 ${HDIR}/main.py --config ${HDIR}/configs/datasets/cifar100/cifar100_ood.yml \
  ${HDIR}/configs/networks/bit.yml \
  ${HDIR}/configs/postprocessors/nak.yml \
  ${HDIR}/configs/pipelines/test/test_ood.yml \
  ${HDIR}/configs/datasets/cifar100/cifar100.yml \
  ${HDIR}/configs/preprocessors/base_preprocessor.yml \
  --network.pretrained True \
  --network.model BiT-M-R50x1 \
  --dataset.pre_size 128 \
  --dataset.image_size 128 \
  --postprocessor.postprocessor_args.jac_chunk_size 4 \
  --network.checkpoint ${HDIR}/results/checkpoints/cifar100_bit.cpt ${@:1}
