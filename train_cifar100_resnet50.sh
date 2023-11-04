#!/bin/bash
HDIR=/h/nng/projects/OpenOOD
python3 ${HDIR}/main.py --config ${HDIR}/configs/datasets/cifar100/cifar100.yml \
    ${HDIR}/configs/pipelines/train/baseline.yml \
    ${HDIR}/configs/networks/resnet50.yml \
    ${HDIR}/configs/preprocessors/base_preprocessor.yml \
    --merge_option merge \
    --optimizer.lr 0.004 \
    --network.pretrained True \
    --network.checkpoint /fs01/home/nng/slurm/2023-10-18/run_bash/cifar100,train/results/cifar100_resnet50_base_e200_lr0.1_default/s0/best_epoch149_acc0.7930.ckpt \
    --optimizer.num_epochs 51 \
    --network.cifar True \
    --seed 0 ${@:1}
