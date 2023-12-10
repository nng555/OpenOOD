#!/bin/bash
HDIR=/h/nng/projects/OpenOOD
python3 ${HDIR}/main.py --config ${HDIR}/configs/datasets/cifar100/cifar100_ood.yml ${HDIR}/configs/networks/resnet18_32x32.yml ${HDIR}/configs/postprocessors/nak.yml ${HDIR}/configs/pipelines/test/test_ood.yml ${HDIR}/configs/datasets/cifar100/cifar100.yml ${HDIR}/configs/preprocessors/base_preprocessor.yml --network.pretrained True --network.residual False --network.checkpoint ${HDIR}/results/checkpoints/cifar100_no_residual.ckpt --postprocessor.postprocessor_args.jac_chunk_size 1 ${@:1}
