#!/bin/bash
HDIR=/h/nng/projects/OpenOOD
python3 ${HDIR}/main.py \
  --config ${HDIR}/configs/datasets/imagenet200/imagenet200_ood.yml \
  ${HDIR}/configs/networks/resnet18_224x224.yml \
  ${HDIR}/configs/postprocessors/nak.yml \
  ${HDIR}/configs/pipelines/test/test_ood.yml \
  ${HDIR}/configs/datasets/imagenet200/imagenet200.yml \
  ${HDIR}/configs/preprocessors/base_preprocessor.yml \
  --postprocessor.postprocessor_args.jac_chunk_size 1 \
  --network.pretrained True \
  --network.checkpoint /fs01/home/nng/projects/OpenOOD/results/imagenet200_resnet18_224x224_base_e90_lr0.1_default/s${1}/best.ckpt ${@:2}
