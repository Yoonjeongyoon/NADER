import os, sys, hashlib
import argparse
import time
import random
from datetime import datetime
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import math
import pdb
import json

if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle

from ModelFactory.register import Registers, import_all_modules_for_register2
from train_utils.log import Log

def calculate_md5(fpath, chunk_size=1024 * 1024):
    md5 = hashlib.md5()
    with open(fpath, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            md5.update(chunk)
    return md5.hexdigest()


def check_md5(fpath, md5, **kwargs):
    return md5 == calculate_md5(fpath, **kwargs)


def check_integrity(fpath, md5=None):
    if not os.path.isfile(fpath):
        return False
    if md5 is None:
        return True
    else:
        return check_md5(fpath, md5)

class ImageNet16(torch.utils.data.Dataset):
    # http://image-net.org/download-images
    # A Downsampled Variant of ImageNet as an Alternative to the CIFAR datasets
    # https://arxiv.org/pdf/1707.08819.pdf

    train_list = [
        ["train_data_batch_1", "27846dcaa50de8e21a7d1a35f30f0e91"],
        ["train_data_batch_2", "c7254a054e0e795c69120a5727050e3f"],
        ["train_data_batch_3", "4333d3df2e5ffb114b05d2ffc19b1e87"],
        ["train_data_batch_4", "1620cdf193304f4a92677b695d70d10f"],
        ["train_data_batch_5", "348b3c2fdbb3940c4e9e834affd3b18d"],
        ["train_data_batch_6", "6e765307c242a1b3d7d5ef9139b48945"],
        ["train_data_batch_7", "564926d8cbf8fc4818ba23d2faac7564"],
        ["train_data_batch_8", "f4755871f718ccb653440b9dd0ebac66"],
        ["train_data_batch_9", "bb6dd660c38c58552125b1a92f86b5d4"],
        ["train_data_batch_10", "8f03f34ac4b42271a294f91bf480f29b"],
    ]
    valid_list = [
        ["val_data", "3410e3017fdaefba8d5073aaa65e4bd6"],
    ]

    def __init__(self, root, train, transform, use_num_of_class_only=None):
        self.root = root
        self.transform = transform
        self.train = train  # training set or valid set
        if not self._check_integrity():
            raise RuntimeError("Dataset not found or corrupted.")

        if self.train:
            downloaded_list = self.train_list
        else:
            downloaded_list = self.valid_list
        self.data = []
        self.targets = []

        # now load the picked numpy arrays
        for i, (file_name, checksum) in enumerate(downloaded_list):
            file_path = os.path.join(self.root, file_name)
            # print ('Load {:}/{:02d}-th : {:}'.format(i, len(downloaded_list), file_path))
            with open(file_path, "rb") as f:
                if sys.version_info[0] == 2:
                    entry = pickle.load(f)
                else:
                    entry = pickle.load(f, encoding="latin1")
                self.data.append(entry["data"])
                self.targets.extend(entry["labels"])
        self.data = np.vstack(self.data).reshape(-1, 3, 16, 16)
        self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC
        if use_num_of_class_only is not None:
            assert (
                isinstance(use_num_of_class_only, int)
                and use_num_of_class_only > 0
                and use_num_of_class_only < 1000
            ), "invalid use_num_of_class_only : {:}".format(use_num_of_class_only)
            new_data, new_targets = [], []
            for I, L in zip(self.data, self.targets):
                if 1 <= L <= use_num_of_class_only:
                    new_data.append(I)
                    new_targets.append(L)
            self.data = new_data
            self.targets = new_targets
        #    self.mean.append(entry['mean'])
        # self.mean = np.vstack(self.mean).reshape(-1, 3, 16, 16)
        # self.mean = np.mean(np.mean(np.mean(self.mean, axis=0), axis=1), axis=1)
        # print ('Mean : {:}'.format(self.mean))
        # temp      = self.data - np.reshape(self.mean, (1, 1, 1, 3))
        # std_data  = np.std(temp, axis=0)
        # std_data  = np.mean(np.mean(std_data, axis=0), axis=0)
        # print ('Std  : {:}'.format(std_data))

    def __repr__(self):
        return "{name}({num} images, {classes} classes)".format(
            name=self.__class__.__name__,
            num=len(self.data),
            classes=len(set(self.targets)),
        )

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index] - 1

        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        return img, target

    def __len__(self):
        return len(self.data)

    def _check_integrity(self):
        root = self.root
        for fentry in self.train_list + self.valid_list:
            filename, md5 = fentry[0], fentry[1]
            fpath = os.path.join(root, filename)
            if not check_integrity(fpath, md5):
                return False
        return True
    
def get_training_dataloader(mean, std, batch_size=16, num_workers=2, shuffle=True):
    train_transform = transforms.Compose([
        transforms.RandomCrop(16, padding=2),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    train_data = ImageNet16('datasets/ImageNet16', True, train_transform, 120)
    assert len(train_data) == 151700
    loader = DataLoader(train_data, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)
    return loader

def get_val_dataloader(mean, std, path='train_utils/imagenet-16-120-test-split.txt', batch_size=16, num_workers=2, shuffle=True):
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    test_data = ImageNet16('datasets/ImageNet16', False, test_transform, 120)
    assert len(test_data) == 6000
    with open(path,'r') as f:
        ds = json.load(f)
        ids = [int(x) for x in ds['xvalid'][1]]
    loader = DataLoader(test_data, sampler=torch.utils.data.sampler.SubsetRandomSampler(ids),num_workers=num_workers, batch_size=batch_size)
    return loader
    
def get_test_dataloader(mean, std, batch_size=16, num_workers=2, shuffle=True):
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    test_data = ImageNet16('datasets/ImageNet16', False, test_transform, 120)
    assert len(test_data) == 6000
    loader = DataLoader(test_data, shuffle=False, num_workers=num_workers, batch_size=batch_size)
    return loader


def train(epoch,log):
    start = time.time()
    net.train()
    for batch_index, (images, labels) in enumerate(training_loader):
        if args.gpu:
            labels = labels.cuda()
            images = images.cuda()
        optimizer.zero_grad()
        outputs = net(images)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()
        n_iter = (epoch - 1) * len(training_loader) + batch_index + 1
        if (batch_index+1)%50==0:
            print('Training Epoch: {epoch} [{trained_samples}/{total_samples}]\tLoss: {:0.4f}\tLR: {:0.6f}'.format(
                loss.item(),
                optimizer.param_groups[0]['lr'],
                epoch=epoch,
                trained_samples=batch_index * args.b + len(images),
                total_samples=len(training_loader.dataset)
            ))
        if (batch_index+1)%10==0:
            log.update('train_loss',loss.item(), n_iter)
    finish = time.time()
    print('epoch {} training time consumed: {:.2f}s'.format(epoch, finish - start))

@torch.no_grad()
def eval_training(loader, tb=True):
    start = time.time()
    net.eval()
    test_loss = 0.0 # cost function error
    correct,total = 0.0,0.0
    for (images, labels) in loader:
        if args.gpu:
            images = images.cuda()
            labels = labels.cuda()
        outputs = net(images)
        loss = loss_function(outputs, labels)
        test_loss += loss.item()
        _, preds = outputs.max(1)
        total += len(outputs)
        correct += preds.eq(labels).sum()
    finish = time.time()
    res = {
        'loss':test_loss / total,
        'acc':correct.float() / total,
        'time':finish - start
    }
    return res

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

if __name__ == '__main__':
    set_seed(888)

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, required=True, help='net type')
    parser.add_argument('-gpu', type=bool, default=True)
    parser.add_argument('-b', type=int, default=256, help='batch size for dataloader')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('-lr', type=float, default=0.1, help='initial learning rate')
    parser.add_argument('-resume', action='store_true', default=False, help='resume training')
    parser.add_argument('--code-dir',default='ModelFactory/codes/nas-bench-201',type=str)
    parser.add_argument('--output', type=str, default='output/nas-bench-201-imagenet16-120')
    parser.add_argument('--tag',default='1', help='tag of experiment')
    args = parser.parse_args()

    import_all_modules_for_register2(args.code_dir)

    log_dir = os.path.join(args.output,args.model_name,args.tag)
    log = Log(log_dir)
    status_path = os.path.join(log_dir,'train_status.txt')
    try:
        net = Registers.model[args.model_name](num_classes=120)
        net = net.cuda()

        #data preprocessing:
        mean = [x / 255 for x in [122.68, 116.66, 104.01]]
        std = [x / 255 for x in [63.22, 61.26, 65.09]]
        training_loader = get_training_dataloader(
            mean,
            std,
            num_workers=4,
            batch_size=args.b,
            shuffle=True
        )
        val_loader = get_val_dataloader(
            mean,
            std,
            num_workers=4,
            batch_size=args.b,
        )
        test_loader = get_test_dataloader(
            mean,
            std,
            num_workers=4,
            batch_size=args.b,
        )

        loss_function = nn.CrossEntropyLoss()
        optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4,nesterov=True)
        train_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)


        best_epoch,best_acc,best_val_acc,best_val_acc = -1,0.0,-1,0.0    
        for epoch in range(1, args.epochs + 1):
            train(epoch,log)
            train_scheduler.step(epoch)
            res = eval_training(val_loader)
            print('Test set: Epoch: {}, Average loss: {:.4f}, Accuracy: {:.4f}, Time consumed:{:.2f}s'.format(
                epoch,
                res['loss'],
                res['acc'],
                res['time']
            ))
            res2 = eval_training(test_loader)
            print('Test set: Epoch: {}, Average loss: {:.4f}, Accuracy: {:.4f}, Time consumed:{:.2f}s'.format(
                epoch,
                res2['loss'],
                res2['acc'],
                res2['time']
            ))
            log.update('val_acc',epoch,res['acc']*100)
            log.update('test_acc',epoch,res2['acc']*100)
            if best_val_acc < res['acc']:
                best_val_acc = res['acc']
                best_val_epoch = epoch
            if best_acc < res2['acc']:
                weights_path = os.path.join(log_dir, 'best.pth')
                print('saving weights file to {}'.format(weights_path))
                torch.save(net.state_dict(), weights_path)
                best_acc = res2['acc']
                best_epoch = epoch
            print(f"val best epoch {best_val_epoch}: acc{best_val_acc}")
            print(f"Test best epoch {best_epoch}: acc{best_acc}")
            if math.isnan(res['loss']) and math.isnan(res2['loss']):
                break
    except Exception as e:
        with open(status_path,'w') as f:
            f.write(f'error:{e}')
    else:
        with open(status_path,'w') as f:
            f.write('success')

