#!/bin/bash
HDIR=/h/nng/projects/OpenOOD
python3 ${HDIR}/main.py --config ${HDIR}/configs/datasets/mnist/mnist_ood.yml ${HDIR}/configs/networks/lenet.yml ${HDIR}/configs/postprocessors/msp.yml ${HDIR}/configs/pipelines/test/test_ood.yml ${HDIR}/configs/datasets/mnist/mnist.yml ${HDIR}/configs/preprocessors/base_preprocessor.yml --network.pretrained True --network.checkpoint ${HDIR}/models/checkpoints/mnist_lenet_acc99.60.ckpt $*
