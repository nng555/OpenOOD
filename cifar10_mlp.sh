#!/bin/bash
HDIR=/h/nng/projects/OpenOOD
python3 ${HDIR}/main.py --config ${HDIR}/configs/datasets/cifar10/cifar10_ood.yml \
  ${HDIR}/configs/networks/mlp.yml \
  ${HDIR}/configs/postprocessors/nak.yml \
  ${HDIR}/configs/pipelines/test/test_ood.yml \
  ${HDIR}/configs/datasets/cifar10/cifar10.yml \
  ${HDIR}/configs/preprocessors/base_preprocessor.yml \
  --network.pretrained True \
  --network.im_size 3072 \
  --network.checkpoint /fs01/home/nng/slurm/2023-11-09/run_bash/cifar10,mlp,train/results/cifar10_mlp_base_e200_lr0.001_default/s0/best.ckpt ${@:1}
