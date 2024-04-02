#!/bin/bash
HDIR=/h/nng/projects/OpenOOD
python3 ${HDIR}/main.py --config ${HDIR}/configs/datasets/cifar10/cifar10_c_idv.yml \
  ${HDIR}/configs/networks/resnet18_32x32.yml \
  ${HDIR}/configs/postprocessors/nak.yml \
  ${HDIR}/configs/pipelines/test/test_ood.yml \
  ${HDIR}/configs/datasets/cifar10/cifar10.yml \
  ${HDIR}/configs/preprocessors/base_preprocessor.yml \
  --network.pretrained True \
  --network.checkpoint ${HDIR}/paper/unc/models/swa_new.ckpt \
  --postprocessor.postprocessor_args.nsteps 5 \
  --postprocessor.postprocessor_args.swag_path ${HDIR}/paper/unc/models/swa_var.ckpt \
  --postprocessor.postprocessor_args.regret step ${@:1}
