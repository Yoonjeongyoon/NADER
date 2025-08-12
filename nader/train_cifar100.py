import os
import sys
import argparse
import time
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import math
import pdb
import json
import signal


from ModelFactory.register import Registers, import_all_modules_for_register2
import train_utils.cifar100_settings as settings
from train_utils.log import Log


def get_training_dataloader(mean, std, batch_size=16, num_workers=2, shuffle=True):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    cifar100_training = torchvision.datasets.CIFAR100(root='datasets', train=True, download=False, transform=transform_train)
    cifar100_training_loader = DataLoader(
        cifar100_training, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)
    return cifar100_training_loader

def get_val_dataloader(mean, std, path='train_utils/cifar100-test-split.txt', batch_size=16, num_workers=2, shuffle=True):
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    cifar100_test = torchvision.datasets.CIFAR100(root='datasets', train=False, download=False, transform=transform_test)
    with open(path,'r') as f:
        ds = json.load(f)
        ids = [int(x) for x in ds['xvalid'][1]]
    cifar100_val_loader = DataLoader(cifar100_test, sampler=torch.utils.data.sampler.SubsetRandomSampler(ids),num_workers=num_workers, batch_size=batch_size)
    return cifar100_val_loader
    
def get_test_dataloader(mean, std, batch_size=16, num_workers=2, shuffle=True):
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    cifar100_test = torchvision.datasets.CIFAR100(root='datasets', train=False, download=False, transform=transform_test)
    cifar100_test_loader = DataLoader(
        cifar100_test, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)
    return cifar100_test_loader


def train(epoch,net,loader,loss_function,optimizer,log):

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

        last_layer = list(net.children())[-1]

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


def main(args,log_dir):
    log = Log(log_dir)

    net = Registers.model[args.model_name](num_classes=1000)
    net = net.cuda()

    #data preprocessing:
    cifar100_training_loader = get_training_dataloader(
        settings.CIFAR100_TRAIN_MEAN,
        settings.CIFAR100_TRAIN_STD,
        num_workers=4,
        batch_size=args.b,
        shuffle=True
    )

    cifar100_val_loader = get_val_dataloader(
        settings.CIFAR100_TRAIN_MEAN,
        settings.CIFAR100_TRAIN_STD,
        num_workers=4,
        batch_size=args.b,
    )

    cifar100_test_loader = get_test_dataloader(
        settings.CIFAR100_TRAIN_MEAN,
        settings.CIFAR100_TRAIN_STD,
        num_workers=4,
        batch_size=args.b,
        shuffle=True
    )

    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4,nesterov=True)
    train_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=settings.EPOCH)



    #use tensorboard
    if not os.path.exists(settings.LOG_DIR):
        os.mkdir(settings.LOG_DIR)


    input_tensor = torch.Tensor(1, 3, 32, 32)
    if args.gpu:
        input_tensor = input_tensor.cuda()

    best_epoch,best_acc,best_val_acc,best_val_acc = -1,0.0,-1,0.0
    
    for epoch in range(1, settings.EPOCH + 1):

        train(epoch,net,cifar100_training_loader,loss_function,optimizer,log)
        train_scheduler.step(epoch)

        res = eval_training(net,cifar100_val_loader,loss_function)
        print('Test set: Epoch: {}, Average loss: {:.4f}, Accuracy: {:.4f}, Time consumed:{:.2f}s'.format(
            epoch,
            res['loss'],
            res['acc'],
            res['time']
        ))

        res2 = eval_training(net,cifar100_test_loader,loss_function)
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

if __name__ == '__main__':
    status_path = 'train_status.txt'
    def handle_signal(signal_number, frame):
        with open(status_path, "w") as f:
            f.write("error")
            sys.exit(1)
    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, required=True, help='net type')
    parser.add_argument('-gpu', type=bool, default=True)
    parser.add_argument('-b', type=int, default=256, help='batch size for dataloader')
    parser.add_argument('-lr', type=float, default=0.1, help='initial learning rate')
    parser.add_argument('-resume', action='store_true', default=False, help='resume training')
    parser.add_argument('--code-dir', type=str)
    parser.add_argument('--output', type=str, default='output/nas-bench-201-cifar100')
    parser.add_argument('--tag', help='tag of experiment')
    args = parser.parse_args()

    import_all_modules_for_register2(args.code_dir)

    log_dir = os.path.join(args.output,args.model_name,args.tag)
    status_path = os.path.join(log_dir,'train_status.txt')

    # try:
    main(args,log_dir)
    # except Exception as e:
    #     with open(status_path,'w') as f:
    #         f.write(f'error:{e}')
    # else:
    #     with open(status_path,'w') as f:
    #         f.write('success')

