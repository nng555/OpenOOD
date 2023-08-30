#!/bin/bash
python3 main.py --config configs/datasets/cifar100/cifar100_ood.yml configs/networks/resnet18_32x32.yml configs/postprocessors/nak.yml configs/pipelines/test/test_ood.yml configs/datasets/cifar100/cifar100.yml configs/preprocessors/base_preprocessor.yml --network.pretrained True --network.checkpoint results/checkpoints/cifar100_resnet18_32x32_base_e100_lr0.1_default/s0/best.ckpt $*
