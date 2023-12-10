#!/bin/bash
HDIR=/h/nng/projects/OpenOOD
python3 ${HDIR}/main.py --config ${HDIR}/configs/datasets/cifar10/cifar10_ood.yml \
  ${HDIR}/configs/networks/vgg.yml \
  ${HDIR}/configs/postprocessors/nak.yml \
  ${HDIR}/configs/pipelines/test/test_ood.yml \
  ${HDIR}/configs/datasets/cifar10/cifar10.yml \
  ${HDIR}/configs/preprocessors/base_preprocessor.yml \
  --network.pretrained True \
  --network.use_bn True \
  --postprocessor.postprocessor_args.jac_chunk_size 4 \
  --network.checkpoint /fs01/home/nng/projects/OpenOOD/results/checkpoints/cifar10_vgg_bn.ckpt  ${@:1}
