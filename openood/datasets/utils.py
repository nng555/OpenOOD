import os
import numpy as np
import json
import torch
from numpy import load
from torch.utils.data import DataLoader, Subset

from openood.preprocessors.test_preprocessor import TestStandardPreProcessor
from openood.preprocessors.utils import get_preprocessor
from openood.utils.config import Config

from .feature_dataset import FeatDataset
from .imglist_dataset import ImglistDataset
from .np_dataset import NPDataset
from .imglist_augmix_dataset import ImglistAugMixDataset
from .imglist_extradata_dataset import ImglistExtraDataDataset, TwoSourceSampler
from .udg_dataset import UDGDataset


def get_dataloader(config: Config):
    # prepare a dataloader dictionary
    dataset_config = config.dataset

    subset = getattr(dataset_config, 'subset', -1)

    dataloader_dict = {}
    for split in dataset_config.split_names:
        split_config = dataset_config[split]
        preprocessor = get_preprocessor(config, split)
        # weak augmentation for data_aux
        data_aux_preprocessor = TestStandardPreProcessor(config)

        if split_config.dataset_class == 'cifar':
            dataset = CIFAR10(
                '/h/nng/projects/OpenOOD/data/images_classic/cifar10',
                train=(split == 'train'),
                transform=preprocessor.transform,
            )

            dataloader = DataLoader(
                dataset,
                num_workers=dataset_config.num_workers,
                shuffle=False,
            )

        if split_config.dataset_class == 'ImglistExtraDataDataset':
            dataset = ImglistExtraDataDataset(
                name=dataset_config.name + '_' + split,
                imglist_pth=split_config.imglist_pth,
                data_dir=split_config.data_dir,
                num_classes=dataset_config.num_classes,
                preprocessor=preprocessor,
                data_aux_preprocessor=data_aux_preprocessor,
                extra_data_pth=split_config.extra_data_pth,
                extra_label_pth=split_config.extra_label_pth,
                extra_percent=split_config.extra_percent)

            batch_sampler = TwoSourceSampler(dataset.orig_ids,
                                             dataset.extra_ids,
                                             split_config.batch_size,
                                             split_config.orig_ratio)

            dataloader = DataLoader(
                dataset,
                batch_sampler=batch_sampler,
                num_workers=dataset_config.num_workers,
            )
        elif split_config.dataset_class == 'ImglistAugMixDataset':
            dataset = ImglistAugMixDataset(
                name=dataset_config.name + '_' + split,
                imglist_pth=split_config.imglist_pth,
                data_dir=split_config.data_dir,
                num_classes=dataset_config.num_classes,
                preprocessor=preprocessor,
                data_aux_preprocessor=data_aux_preprocessor)
            sampler = None
            if dataset_config.num_gpus * dataset_config.num_machines > 1:
                sampler = torch.utils.data.distributed.DistributedSampler(
                    dataset)
                split_config.shuffle = False

            dataloader = DataLoader(dataset,
                                    batch_size=split_config.batch_size,
                                    shuffle=split_config.shuffle,
                                    num_workers=dataset_config.num_workers,
                                    sampler=sampler)
        else:
            CustomDataset = eval(split_config.dataset_class)
            dataset = CustomDataset(
                name=dataset_config.name + '_' + split,
                imglist_pth=split_config.imglist_pth,
                data_dir=split_config.data_dir,
                num_classes=dataset_config.num_classes,
                preprocessor=preprocessor,
                data_aux_preprocessor=data_aux_preprocessor)
            sampler = None

            if subset > 0 and split != 'train' and split != 'val':
                nskip = int(1. / subset)
                dataset = Subset(dataset, list(range(0, len(dataset), nskip)))
            #if dataset_config.num_gpus * dataset_config.num_machines > 1:
            #    sampler = torch.utils.data.distributed.DistributedSampler(
            #        dataset)
            #    split_config.shuffle = False

            dataloader = DataLoader(dataset,
                                    batch_size=split_config.batch_size,
                                    shuffle=split_config.shuffle,
                                    num_workers=dataset_config.num_workers,
                                    sampler=sampler)
            """
            if split_config.get('class_loader', False):
                class_loaders = []
                for k in range(dataset_config.num_classes):
                    k_idxs = [i for i, l in enumerate(dataset.labels) if l == k]
                    k_dataset = Subset(dataset, k_idxs)
                    class_loaders.append(DataLoader(
                        k_dataset,
                        batch_size=split_config.batch_size,
                        shuffle=split_config.shuffle,
                        num_workers=dataset_config.num_workers,
                    ))
                dataloader_dict[split + '_class'] = class_loaders
            """

        dataloader_dict[split] = dataloader

    if 'corruption_path' in config.dataset and config.dataset.corruption_path != 'default':
        print(f"Corrupting train data from {config.dataset.corruption_path}")
        cpath = '/'.join(config.dataset.train.imglist_pth.split('/')[:-1]) + f'/train_{config.dataset.name}_{config.dataset.corruption_path}.npy'
        corruptions = np.load(cpath, allow_pickle=True).item()['cmap']

        print("Example corruptions...")
        i = 0
        for orig, new in corruptions.items():
            print(f"{orig.strip()} --> {new.strip()}")
            i += 1
            if i == 3:
                break

        dataloader_dict['train'].dataset.imglist = [corruptions.get(img, img) for img in dataloader_dict['train'].dataset.imglist]

    if 'prune' in config.dataset and config.dataset.prune != 0:
        assert os.path.exists(config.dataset.prune_path), "Pruning order not found"
        # assume ordered from least useful to most useful
        prune_order = np.load(config.dataset.prune_path)
        prune_idx = int(config.dataset.prune * len(prune_order))
        print(f"Pruning {prune_idx} examples")
        prune_order = np.sort(prune_order[prune_idx:])
        dataloader_dict['train'].dataset.imglist = [dataloader_dict['train'].dataset.imglist[i] for i in prune_order]

    return dataloader_dict


def get_ood_dataloader(config: Config):
    # specify custom dataset class
    ood_config = config.ood_dataset
    subset = getattr(ood_config, 'subset', -1)
    CustomDataset = eval(ood_config.dataset_class)
    dataloader_dict = {}
    for split in ood_config.split_names:
        split_config = ood_config[split]
        preprocessor = get_preprocessor(config, split)
        data_aux_preprocessor = TestStandardPreProcessor(config)
        if split == 'val':
            # validation set
            dataset = CustomDataset(
                name=ood_config.name + '_' + split,
                imglist_pth=split_config.imglist_pth,
                data_dir=split_config.data_dir,
                num_classes=ood_config.num_classes,
                preprocessor=preprocessor,
                data_aux_preprocessor=data_aux_preprocessor)
            if subset > 0:
                nskip = int(1. / subset)
                dataset = Subset(dataset, list(range(0, len(dataset), nskip)))
            dataloader = DataLoader(dataset,
                                    batch_size=ood_config.batch_size,
                                    shuffle=ood_config.shuffle,
                                    num_workers=ood_config.num_workers)
            dataloader_dict[split] = dataloader
        else:
            # dataloaders for csid, nearood, farood
            sub_dataloader_dict = {}

            if isinstance(split_config.datasets, str):
                split_config.datasets = [split_config.datasets]

            for dataset_name in split_config.datasets:
                if 'np' in split_config and split_config.np:
                    dataset = NPDataset(
                        name=ood_config.name + '_' + split,
                        data_dir=split_config.data_dir,
                        corruption=dataset_name,
                        label_pth=split_config.label_pth,
                        num_classes=ood_config.num_classes,
                        preprocessor=preprocessor,
                        data_aux_preprocessor=data_aux_preprocessor)
                else:
                    dataset_config = split_config[dataset_name]
                    dataset = CustomDataset(
                        name=ood_config.name + '_' + split,
                        imglist_pth=dataset_config.imglist_pth,
                        data_dir=dataset_config.data_dir,
                        num_classes=ood_config.num_classes,
                        preprocessor=preprocessor,
                        data_aux_preprocessor=data_aux_preprocessor)
                if subset > 0:
                    nskip = int(1. / subset)
                    dataset = Subset(dataset, list(range(0, len(dataset), nskip)))
                dataloader = DataLoader(dataset,
                                        batch_size=ood_config.batch_size,
                                        shuffle=ood_config.shuffle,
                                        num_workers=ood_config.num_workers)
                sub_dataloader_dict[dataset_name] = dataloader
            dataloader_dict[split] = sub_dataloader_dict

    return dataloader_dict


def get_feature_dataloader(dataset_config: Config):
    # load in the cached feature
    loaded_data = load(dataset_config.feat_path, allow_pickle=True)
    total_feat = torch.from_numpy(loaded_data['feat_list'])
    del loaded_data
    # reshape the vector to fit in to the network
    total_feat.unsqueeze_(-1).unsqueeze_(-1)
    # let's see what we got here should be something like:
    # torch.Size([total_num, channel_size, 1, 1])
    print('Loaded feature size: {}'.format(total_feat.shape))

    split_config = dataset_config['train']

    dataset = FeatDataset(feat=total_feat)
    dataloader = DataLoader(dataset,
                            batch_size=split_config.batch_size,
                            shuffle=split_config.shuffle,
                            num_workers=dataset_config.num_workers)

    return dataloader


def get_feature_opengan_dataloader(dataset_config: Config):
    feat_root = dataset_config.feat_root

    dataloader_dict = {}
    for d in ['id_train', 'id_val', 'ood_val']:
        # load in the cached feature
        loaded_data = load(os.path.join(feat_root, f'{d}.npz'),
                           allow_pickle=True)
        total_feat = torch.from_numpy(loaded_data['feat_list'])
        total_labels = loaded_data['label_list']
        del loaded_data
        # reshape the vector to fit in to the network
        total_feat.unsqueeze_(-1).unsqueeze_(-1)
        # let's see what we got here should be something like:
        # torch.Size([total_num, channel_size, 1, 1])
        print('Loaded feature size: {}'.format(total_feat.shape))

        if d == 'id_train':
            split_config = dataset_config['train']
        else:
            split_config = dataset_config['val']

        dataset = FeatDataset(feat=total_feat, labels=total_labels)
        dataloader = DataLoader(dataset,
                                batch_size=split_config.batch_size,
                                shuffle=split_config.shuffle,
                                num_workers=dataset_config.num_workers)
        dataloader_dict[d] = dataloader

    return dataloader_dict
