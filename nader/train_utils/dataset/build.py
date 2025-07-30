# --------------------------------------------------------
# InternImage
# Copyright (c) 2022 OpenGVLab
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

import os
import torch
import numpy as np
import torch.distributed as dist
from torch import FloatTensor, div
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from timm.data import Mixup
from timm.data import create_transform
from .cached_image_folder import ImageCephDataset
from .samplers import SubsetRandomSampler, NodeDistributedSampler
import pickle

try:
    from torchvision.transforms import InterpolationMode

    def _pil_interp(method):
        if method == 'bicubic':
            return InterpolationMode.BICUBIC
        elif method == 'lanczos':
            return InterpolationMode.LANCZOS
        elif method == 'hamming':
            return InterpolationMode.HAMMING
        else:
            return InterpolationMode.BILINEAR
except:
    from timm.data.transforms import _pil_interp

class CustomImageDataset(Dataset):

    def __init__(self, img_dir, txt_file, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.img_labels = []
        with open(txt_file, 'r') as f:
            for line in f:
                path, label = line.strip().split(',')
                self.img_labels.append((path, int(label)))  # 标签转为整数

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path, label = self.img_labels[idx]
        full_path = os.path.join(self.img_dir, img_path)
        image = Image.open(full_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


class TinyImageNetDataset(Dataset):
    """Dataset class for ImageNet"""
    def __init__(self, dataset, labels, transform=None, normalize=None):
        super(TinyImageNetDataset, self).__init__()
        assert(len(dataset) == len(labels))
        self.dataset = dataset
        self.labels = labels
        self.transform = transform
        self.normalize = normalize

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data = self.dataset[idx]
        if self.transform:
            data = self.transform(data)

        data = div(data.type(FloatTensor), 255)
        if self.normalize:
            data = self.normalize(data)

        return data, self.labels[idx]

class TTA(torch.nn.Module):

    def __init__(self, size, scales=[1.0, 1.05, 1.1]):
        super().__init__()
        self.size = size
        self.scales = scales

    def forward(self, img):
        out = []
        cc = transforms.CenterCrop(self.size)
        for scale in self.scales:
            size_ = int(scale * self.size)
            rs = transforms.Resize(size_, interpolation=_pil_interp('bicubic'))
            img_ = rs(img)
            img_ = cc(img_)
            out.append(img_)

        return out

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(size={self.size}, scale={self.scales})"


def build_loader(config):
    config.defrost()
    dataset_train, config.MODEL.NUM_CLASSES = build_dataset('train',
                                                            config=config)
    config.freeze()
    dataset_val, _ = build_dataset('val', config=config)
    dataset_test, _ = build_dataset('val', config=config)
    
    num_tasks = dist.get_world_size()
    global_rank = dist.get_rank()
    if dataset_train is not None:
        if config.DATA.IMG_ON_MEMORY:
            sampler_train = NodeDistributedSampler(dataset_train)
        else:
            if config.DATA.ZIP_MODE and config.DATA.CACHE_MODE == 'part':
                indices = np.arange(dist.get_rank(), len(dataset_train),
                                    dist.get_world_size())
                sampler_train = SubsetRandomSampler(indices)
            else:
                sampler_train = torch.utils.data.DistributedSampler(
                    dataset_train,
                    num_replicas=num_tasks,
                    rank=global_rank,
                    shuffle=True)

    if dataset_val is not None:
        if config.TEST.SEQUENTIAL:
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)
        else:
            sampler_val = torch.utils.data.distributed.DistributedSampler(
                dataset_val, shuffle=False)

    if dataset_test is not None:
        if config.TEST.SEQUENTIAL:
            sampler_test = torch.utils.data.SequentialSampler(dataset_test)
        else:
            sampler_test = torch.utils.data.distributed.DistributedSampler(
                dataset_test, shuffle=False)

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        sampler=sampler_train,
        batch_size=config.DATA.BATCH_SIZE,
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=config.DATA.PIN_MEMORY,
        drop_last=True,
        persistent_workers=True) if dataset_train is not None else None

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val,
        sampler=sampler_val,
        batch_size=config.DATA.BATCH_SIZE,
        shuffle=False,
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=config.DATA.PIN_MEMORY,
        drop_last=False,
        persistent_workers=True) if dataset_val is not None else None

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test,
        sampler=sampler_test,
        batch_size=config.DATA.BATCH_SIZE,
        shuffle=False,
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=config.DATA.PIN_MEMORY,
        drop_last=False,
        persistent_workers=True) if dataset_test is not None else None

    # setup mixup / cutmix
    mixup_fn = None
    mixup_active = config.AUG.MIXUP > 0 or config.AUG.CUTMIX > 0. or config.AUG.CUTMIX_MINMAX is not None
    if mixup_active:
        mixup_fn = Mixup(mixup_alpha=config.AUG.MIXUP,
                         cutmix_alpha=config.AUG.CUTMIX,
                         cutmix_minmax=config.AUG.CUTMIX_MINMAX,
                         prob=config.AUG.MIXUP_PROB,
                         switch_prob=config.AUG.MIXUP_SWITCH_PROB,
                         mode=config.AUG.MIXUP_MODE,
                         label_smoothing=config.MODEL.LABEL_SMOOTHING,
                         num_classes=config.MODEL.NUM_CLASSES)

    return dataset_train, dataset_val, dataset_test, data_loader_train, \
        data_loader_val, data_loader_test, mixup_fn


def build_loader2(config):
    config.defrost()
    dataset_train, config.MODEL.NUM_CLASSES = build_dataset('train',
                                                            config=config)
    config.freeze()
    dataset_val, _ = build_dataset('val', config=config)
    dataset_test, _ = build_dataset('val', config=config)

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        shuffle=True,
        batch_size=config.DATA.BATCH_SIZE,
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=config.DATA.PIN_MEMORY,
        drop_last=True,
        persistent_workers=True) if dataset_train is not None else None

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=config.DATA.BATCH_SIZE,
        shuffle=False,
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=config.DATA.PIN_MEMORY,
        drop_last=False,
        persistent_workers=True) if dataset_val is not None else None

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=config.DATA.BATCH_SIZE,
        shuffle=False,
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=config.DATA.PIN_MEMORY,
        drop_last=False,
        persistent_workers=True) if dataset_test is not None else None

    # setup mixup / cutmix
    mixup_fn = None
    mixup_active = config.AUG.MIXUP > 0 or config.AUG.CUTMIX > 0. or config.AUG.CUTMIX_MINMAX is not None
    if mixup_active:
        mixup_fn = Mixup(mixup_alpha=config.AUG.MIXUP,
                         cutmix_alpha=config.AUG.CUTMIX,
                         cutmix_minmax=config.AUG.CUTMIX_MINMAX,
                         prob=config.AUG.MIXUP_PROB,
                         switch_prob=config.AUG.MIXUP_SWITCH_PROB,
                         mode=config.AUG.MIXUP_MODE,
                         label_smoothing=config.MODEL.LABEL_SMOOTHING,
                         num_classes=config.MODEL.NUM_CLASSES)

    return dataset_train, dataset_val, dataset_test, data_loader_train, \
        data_loader_val, data_loader_test, mixup_fn


def build_dataset(split, config):
    transform = build_transform(split == 'train', config)
    dataset = None
    nb_classes = None
    prefix = split
    if config.DATA.DATASET == 'imagenet':
        if prefix == 'train' and not config.EVAL_MODE:
            root = os.path.join(config.DATA.DATA_PATH, 'train')
            dataset = ImageCephDataset(root,
                                       'train',
                                       transform=transform,
                                       on_memory=config.DATA.IMG_ON_MEMORY)
        elif prefix == 'val':
            root = os.path.join(config.DATA.DATA_PATH, 'val')
            dataset = ImageCephDataset(root, 'val', transform=transform)
        elif prefix == 'test':
            root = os.path.join(config.DATA.DATA_PATH,'val')
            dataset = ImageCephDataset(root,'val',transform=transform)
        nb_classes = 1000
    elif config.DATA.DATASET == 'imagenet22K':
        if prefix == 'train':
            if not config.EVAL_MODE:
                root = config.DATA.DATA_PATH
                dataset = ImageCephDataset(root,
                                           'train',
                                           transform=transform,
                                           on_memory=config.DATA.IMG_ON_MEMORY)
            nb_classes = 21841
        elif prefix == 'val':
            root = os.path.join(config.DATA.DATA_PATH, 'val')
            dataset = ImageCephDataset(root, 'val', transform=transform)
            nb_classes = 1000
    elif config.DATA.DATASET == 'tiny_imagenet':
        if prefix == 'train' and not config.EVAL_MODE:
            with open('/mnt/mobile/yangzekang/ModelGen/TinyImageNet-Transformers/train_dataset.pkl', 'rb') as f:
                train_data, train_labels = pickle.load(f)
            transform = transforms.Compose([
                transforms.Resize(config.DATA.IMG_SIZE, interpolation=InterpolationMode.BICUBIC),
                transforms.RandAugment(num_ops=2,magnitude=9),
            ])
            dataset = TinyImageNetDataset(train_data, train_labels.type(torch.LongTensor), transform,
                normalize=transforms.Compose([
                    transforms.Normalize(
                        mean=(0.485, 0.456, 0.406),
                        std=(0.229, 0.224, 0.225)
                    )
                ]),
            )
            f.close()
        elif prefix == 'val':
            with open('/mnt/mobile/yangzekang/ModelGen/TinyImageNet-Transformers/val_dataset.pkl', 'rb') as f:
                val_data, val_labels = pickle.load(f)
            transform = transforms.Compose([
                transforms.Resize(config.DATA.IMG_SIZE, interpolation=InterpolationMode.BICUBIC),
            ])
            dataset = TinyImageNetDataset(val_data, val_labels.type(torch.LongTensor), transform,
                normalize=transforms.Compose([
                    transforms.Normalize(
                        mean=(0.485, 0.456, 0.406),
                        std=(0.229, 0.224, 0.225)
                    ),
                ])
            )
        nb_classes = 200
    elif config.DATA.DATASET == 'cassava_leaf_diease':
        image_dir = '/home/mnt/yangzekang/ModelGen/datasets/CassavaLeafDisease/train_images'
        if prefix == 'train' and not config.EVAL_MODE:
            txt_path = '/home/mnt/yangzekang/ModelGen/datasets/CassavaLeafDisease/train.txt'
            dataset = CustomImageDataset(image_dir, txt_path, transform=transform)
        elif prefix == 'val':
            txt_path = '/home/mnt/yangzekang/ModelGen/datasets/CassavaLeafDisease/test.txt'
            dataset = CustomImageDataset(image_dir, txt_path, transform=transform)
        elif prefix == 'test':
            txt_path = '/home/mnt/yangzekang/ModelGen/datasets/CassavaLeafDisease/test.txt'
            dataset = CustomImageDataset(image_dir, txt_path, transform=transform)
        nb_classes = 5
    else:
        raise NotImplementedError(
            f'build_dataset does support {config.DATA.DATASET}')

    return dataset, nb_classes


def build_transform(is_train, config):
    resize_im = config.DATA.IMG_SIZE > 32
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=config.DATA.IMG_SIZE,
            is_training=True,
            color_jitter=config.AUG.COLOR_JITTER
            if config.AUG.COLOR_JITTER > 0 else None,
            auto_augment=config.AUG.AUTO_AUGMENT
            if config.AUG.AUTO_AUGMENT != 'none' else None,
            re_prob=config.AUG.REPROB,
            re_mode=config.AUG.REMODE,
            re_count=config.AUG.RECOUNT,
            interpolation=config.DATA.INTERPOLATION,
        )
        if not resize_im:
            # replace RandomResizedCropAndInterpolation with
            # RandomCrop
            transform.transforms[0] = transforms.RandomCrop(
                config.DATA.IMG_SIZE, padding=4)

        return transform

    t = []
    if resize_im:
        if config.TEST.CROP:
            size = int(1.0 * config.DATA.IMG_SIZE)
            t.append(
                transforms.Resize(size,
                                  interpolation=_pil_interp(
                                      config.DATA.INTERPOLATION)),
                # to maintain same ratio w.r.t. 224 images
            )
            t.append(transforms.CenterCrop(config.DATA.IMG_SIZE))
        elif config.AUG.RANDOM_RESIZED_CROP:
            t.append(
                transforms.RandomResizedCrop(
                    (config.DATA.IMG_SIZE, config.DATA.IMG_SIZE),
                    interpolation=_pil_interp(config.DATA.INTERPOLATION)))
        else:
            t.append(
                transforms.Resize(
                    (config.DATA.IMG_SIZE, config.DATA.IMG_SIZE),
                    interpolation=_pil_interp(config.DATA.INTERPOLATION)))
    if config.DATA.DATASET != 'tiny_imagenet':
        t.append(transforms.ToTensor())
    t.append(transforms.Normalize(config.AUG.MEAN, config.AUG.STD))

    return transforms.Compose(t)
