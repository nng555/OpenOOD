import time
from tqdm import tqdm

from openood.datasets import get_dataloader, get_ood_dataloader
from openood.evaluators import get_evaluator
from openood.networks import get_network
from openood.postprocessors import get_postprocessor, BasePostprocessor
from openood.utils import setup_logger


class TestOODPipeline:
    def __init__(self, config) -> None:
        self.config = config

    def run(self):
        # generate output directory and save the full config file
        setup_logger(self.config)

        # get dataloader
        if 'use_train' in self.config.postprocessor and self.config.postprocessor.use_train:
            self.config.dataset.train.shuffle = False
            print("Flipping train shuffle to false")

        id_loader_dict = get_dataloader(self.config)
        ood_loader_dict = get_ood_dataloader(self.config)

        # init network
        net = get_network(self.config.network)

        if 'swag' in self.config.network and self.config.network.swag:
            print("Recalibrating batchnorm")
            net.train()
            for batch in tqdm(id_loader_dict['train']):
                out = net(batch['data'].cuda())
            print("Done recalibrating batchnorm")
            net.eval()
            import ipdb; ipdb.set_trace()

        # init ood evaluator
        evaluator = get_evaluator(self.config)

        # init ood postprocessor
        postprocessor = get_postprocessor(self.config)
        # setup for distance-based methods
        postprocessor.setup(net, id_loader_dict, ood_loader_dict)
        print('\n', flush=True)
        print(u'\u2500' * 70, flush=True)

        # start calculating accuracy
        base_postprocessor = BasePostprocessor(None)
        print('\nStart evaluation...', flush=True)
        timer = time.time()
        if self.config.evaluator.ood_scheme == 'fsood':
            acc_metrics = evaluator.eval_acc(
                net,
                id_loader_dict['test'],
                base_postprocessor,
                fsood=True,
                csid_data_loaders=ood_loader_dict['csid'])
        else:
            acc_metrics = evaluator.eval_acc(net, id_loader_dict['val'],
                                             base_postprocessor)
        print('Time used for eval accuracy: {:.4f}s'.format(time.time() - timer))

        print('\nAccuracy {:.2f}%'.format(100 * acc_metrics['acc']),
              flush=True)
        print(u'\u2500' * 70, flush=True)

        # start evaluating ood detection methods
        timer = time.time()
        if self.config.evaluator.ood_scheme == 'fsood':
            evaluator.eval_ood(net,
                               id_loader_dict,
                               ood_loader_dict,
                               postprocessor,
                               fsood=True)
        else:
            evaluator.eval_ood(net, id_loader_dict, ood_loader_dict,
                               postprocessor)
        print('Time used for eval_ood: {:.0f}s'.format(time.time() - timer))
        print('Completed!', flush=True)
