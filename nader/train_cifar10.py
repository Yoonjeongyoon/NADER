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
    
def get_dataloader1(mean, std, batch_size=16, num_workers=2, shuffle=True):
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    train_data = torchvision.datasets.CIFAR10('datasets', train=True, transform=train_transform, download=False)
    test_data = torchvision.datasets.CIFAR10('datasets', train=False, transform=test_transform, download=False)
    assert len(train_data) == 50000 and len(test_data) == 10000
    train_loader = DataLoader(train_data, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)
    test_loader = DataLoader(test_data, shuffle=False, num_workers=num_workers, batch_size=batch_size)
    return train_loader,test_loader

def get_dataloader2(mean, std, path='train_utils/cifar-split.txt', batch_size=16, num_workers=2, shuffle=True):
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    train_data = torchvision.datasets.CIFAR10('datasets', train=True, transform=train_transform, download=False)
    assert len(train_data) == 50000
    with open(path,'r') as f:
        ds = json.load(f)
        ids_train = [int(x) for x in ds['train'][1]]
        ids_test = [int(x) for x in ds['valid'][1]]
    train_loader = DataLoader(train_data, sampler=torch.utils.data.sampler.SubsetRandomSampler(ids_train),num_workers=num_workers, batch_size=batch_size)
    test_loader = DataLoader(train_data, sampler=torch.utils.data.sampler.SubsetRandomSampler(ids_test),num_workers=num_workers, batch_size=batch_size)
    return train_loader,test_loader

def train_one_epoch(net,loader,optimizer,loss_function,train_scheduler,epoch,log):
    start = time.time()
    net.train()
    for batch_index, (images, labels) in enumerate(loader):
        if args.gpu:
            labels = labels.cuda()
            images = images.cuda()
        optimizer.zero_grad()
        outputs = net(images)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()
        n_iter = (epoch - 1) * len(loader) + batch_index + 1
        if (batch_index+1)%50==0:
            print('Training Epoch: {epoch} [{trained_samples}/{total_samples}]\tLoss: {:0.4f}\tLR: {:0.6f}'.format(
                loss.item(),
                optimizer.param_groups[0]['lr'],
                epoch=epoch,
                trained_samples=batch_index * args.b + len(images),
                total_samples=len(loader.dataset)
            ))
        if (batch_index+1)%10==0:
            log.update('train_loss',loss.item(), n_iter)
    finish = time.time()
    print('epoch {} training time consumed: {:.2f}s'.format(epoch, finish - start))

@torch.no_grad()
def eval_training(net,loader,loss_function,tb=True):
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


def train(train_loader,test_loader,typ='val'):
    net = Registers.model[args.model_name](num_classes=10)
    net = net.cuda()

    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4,nesterov=True)
    train_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)


    best_epoch,best_acc,best_val_acc,best_val_acc = -1,0.0,-1,0.0    
    for epoch in range(1, args.epochs + 1):
        train_one_epoch(net,train_loader,optimizer,loss_function,train_scheduler,epoch,log)
        train_scheduler.step(epoch)
        res = eval_training(net,test_loader,loss_function)
        print('Test set: Epoch: {}, Average loss: {:.4f}, Accuracy: {:.4f}, Time consumed:{:.2f}s'.format(
            epoch,
            res['loss'],
            res['acc'],
            res['time']
        ))
        log.update(f'{typ}_acc',epoch,res['acc']*100)
        if best_acc < res['acc']:
            weights_path = os.path.join(log_dir, 'best.pth')
            print('saving weights file to {}'.format(weights_path))
            torch.save(net.state_dict(), weights_path)
            best_acc = res['acc']
            best_epoch = epoch
        print(f"{typ} best epoch {best_epoch}: acc{best_acc}")
        if math.isnan(res['loss']):
            return -1,-1
    return best_epoch,best_acc

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
    parser.add_argument('--output', type=str, default='output/nas-bench-201-cifar10')
    parser.add_argument('--tag',default='1', help='tag of experiment')
    args = parser.parse_args()


    import_all_modules_for_register2(args.code_dir)

    log_dir = os.path.join(args.output,args.model_name,args.tag)
    status_path = os.path.join(log_dir,'train_status.txt')
    log = Log(log_dir)
    try:
        #data preprocessing:
        mean = [x / 255 for x in [125.3, 123.0, 113.9]]
        std = [x / 255 for x in [63.0, 62.1, 66.7]]
        training_loader1,test_loader1 = get_dataloader1(
            mean,
            std,
            num_workers=4,
            batch_size=args.b,
        )
        training_loader2,test_loader2 = get_dataloader2(
            mean,
            std,
            num_workers=4,
            batch_size=args.b,
        )
        best_val = train(training_loader2,test_loader2,'val')
        if best_val[0]==-1:
            log.update(f'test_acc',1,0)
            best_test = [-1,-1]
        else:
            best_test = train(training_loader1,test_loader1,'test')

        print(f"val best epoch {best_val[0]}: acc{best_val[1]}")
        print(f"Test best epoch {best_test[0]}: acc{best_test[1]}")
    except:
        with open(status_path,'w') as f:
            f.write('success')
    else:
        with open(status_path,'w') as f:
            f.write('error')