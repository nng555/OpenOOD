# necessary imports
import torch
import os

from openood.evaluation_api import Evaluator
from openood.networks import ResNet18_32x32 # just a wrapper around the ResNet

# load the model
#net = ResNet18_32x32(num_classes=100)
#base_dir = '/h/nng/projects/jax_ek/experiments/cifar100/cifar100_resnet18_32x32_base_e100_lr0.1_default'
net = ResNet18_32x32(num_classes=10)
base_dir = '/h/nng/projects/jax_ek/experiments/cifar10/cifar10_resnet18_32x32_base_e100_lr0.1_default'
net.load_state_dict(
    torch.load(os.path.join(base_dir, 's0/best.ckpt'))
)
net.cuda()
net.eval();

postprocessor_name = "nak"

evaluator = Evaluator(
    net,
    id_name='cifar10',                     # the target ID dataset
    data_root='./data',                    # change if necessary
    config_root='./configs',                      # see notes above
    preprocessor=None,                     # default preprocessing for the target ID dataset
    postprocessor_name=postprocessor_name, # the postprocessor to use
    batch_size=200,                        # for certain methods the results can be slightly affected by batch size
    shuffle=False,
    num_workers=8)                         # could use more num_workers outside colab

metrics = evaluator.eval_ood(fsood=False)
