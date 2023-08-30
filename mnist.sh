#!/bin/bash
python3 main.py --config configs/datasets/mnist/mnist_ood.yml configs/networks/lenet.yml configs/postprocessors/nak.yml configs/pipelines/test/test_ood.yml configs/datasets/mnist/mnist.yml configs/preprocessors/base_preprocessor.yml --network.pretrained True --network.checkpoint results/checkpoints/mnist_lenet_acc99.60.ckpt $*
