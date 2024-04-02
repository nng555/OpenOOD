#!/bin/bash
HDIR=/h/nng/projects/OpenOOD
python3 ${HDIR}/main.py --config ${HDIR}/configs/datasets/mnist/mnist_ood.yml ${HDIR}/configs/networks/lenet.yml ${HDIR}/configs/postprocessors/nak.yml ${HDIR}/configs/pipelines/test/test_ood.yml ${HDIR}/configs/datasets/mnist/mnist.yml ${HDIR}/configs/preprocessors/base_preprocessor.yml --network.pretrained True --network.checkpoint /h/nng/slurm/2024-01-23/run_bash/grand,mnist,seed_${1},train/last.ckpt --postprocessor.postprocessor_args.natural False ${@:2}
