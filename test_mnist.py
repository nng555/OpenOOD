# necessary imports
import torch
import os

from openood.evaluation_api import Evaluator
from openood.networks.lenet import LeNet # just a wrapper around the ResNet

# load the model
#net = ResNet18_32x32(num_classes=100)
#base_dir = '/h/nng/projects/jax_ek/experiments/cifar100/cifar100_resnet18_32x32_base_e100_lr0.1_default'
net = LeNet(num_classes=10)
base_dir = '/h/nng/projects/OpenOOD/models/checkpoints/mnist_lenet_acc99.60.ckpt'
net.load_state_dict(
    torch.load(base_dir)
)
net.cuda()
net.eval();

postprocessor_name = "nak"

evaluator = Evaluator(
    net,
    id_name='mnist',                     # the target ID dataset
    data_root='./data',                    # change if necessary
    config_root='./configs',                      # see notes above
    preprocessor=None,                     # default preprocessing for the target ID dataset
    postprocessor_name=postprocessor_name, # the postprocessor to use
    batch_size=200,                        # for certain methods the results can be slightly affected by batch size
    shuffle=False,
    num_workers=8)                         # could use more num_workers outside colab

metrics = evaluator.eval_ood(fsood=False)
