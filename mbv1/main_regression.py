from __future__ import print_function
import os
import argparse
import shutil
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
from tqdm import tqdm
from torchvision import datasets, transforms
from torch.autograd import Variable

import sys
from compute_flops import print_model_param_nums, print_model_param_flops

import json

# Training settings
parser = argparse.ArgumentParser(description='PyTorch Slimming CIFAR training')
parser.add_argument('--dataset', type=str, default='cifar10',
                    help='training dataset (default: cifar100)', required=True)
parser.add_argument('--sr', action='store_true', default=False,
                    help='training with sparsity regularization')
parser.add_argument('--sp', action='store_true', default=True,
                    help='with splitting aware model')
parser.add_argument('--s', type=float, default=0.0001,
                    help='scale sparse rate (default: 0.0001)')
parser.add_argument('--batch-size', type=int, default=256, metavar='N',
                    help='input batch size for training (default: 256)')
parser.add_argument('--test-batch-size', type=int, default=100, metavar='N',
                    help='input batch size for testing (default: 100)')
parser.add_argument('--epochs', type=int, default=160, metavar='N',
                    help='number of epochs to train (default: 160)')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                    help='learning rate (default: 0.1)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.9)')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--save', default='./saved_models', type=str, metavar='PATH',
                    help='path to save prune model (default: current directory)')

### Az
parser.add_argument('--custom_model', action='store_true', default=False,
                    help='use custom model')
parser.add_argument('--regression', action='store_true', default=False,
                    help='use regression instead of classification')
parser.add_argument('--input-dim', type=int, default=1,
                    help='input dimension for regression')
parser.add_argument('--output-dim', type=int, default=1,
                    help='output dimension for regression')
parser.add_argument('--hidden-dim', type=int, default=3,
                    help='hidden dimension for regression model')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device('cuda') if args.cuda else torch.device('cpu')

print(args)

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
    
if args.custom_model:
    from model_custom import SimpleModel as model
    
    logging_file = f'{args.dataset}.log'
    model_save_path = f'{args.dataset}_0.pth.tar'

else:
    if args.sp:
        from sp_mbnet import sp_mbnet as mbnet
        from sp_mbnet import splitcfg
    else:
        from mobilenetv1 import MobileNetV1 as mbnet
        from mobilenetv1 import MbBlock, ConvBlock

    assert not (args.sr and args.sp)
    
    logging_file = '{}.log'.format(args.dataset)
    model_save_path = '{}_0.pth.tar'.format(args.dataset)

if not os.path.exists(args.save):
    os.makedirs(args.save)

#########################################################
# create file handler which logs even debug messages
import logging
log = logging.getLogger()
log.setLevel(logging.INFO)

ch = logging.StreamHandler()
fh = logging.FileHandler(os.path.join(args.save, logging_file))
formatter = logging.Formatter('%(asctime)s - %(message)s')
ch.setFormatter(formatter)
fh.setFormatter(formatter)

log.addHandler(fh)
log.addHandler(ch)
#########################################################

# Get data loaders
if args.regression:
    # Custom regression data loader
    try:
        from regression_dataloader import get_dataloader
        train_loader, test_loader = get_dataloader(
            args.dataset,
            train_batch_size=args.batch_size,
            test_batch_size=args.test_batch_size,
            use_cuda=args.cuda,
            input_dim=args.input_dim,
            output_dim=args.output_dim,
            normalize=False,      # DON'T normalize inputs to [0,1]
            standardize=True      # ONLY standardize outputs
        )
    except ImportError:
        NotImplementedError("Regression dataloader not found. Please ensure regression_dataloader.py is present.")
        
else:
    from dataloader import get_data_loader
    train_loader, test_loader = get_data_loader(
        args.dataset,
        train_batch_size=args.batch_size,
        test_batch_size=args.test_batch_size,
        use_cuda=args.cuda
    )

# Create model
if args.custom_model:
    if args.regression:
        # Build cfg for regression
        cfg = [(args.input_dim, args.hidden_dim)]
        print(f"Model config: {cfg}")
    else:
        cfg = None
    model = model(cfg=cfg, dataset=args.dataset, activation='relu')
    print("Model details:", model)
    model.to(device)
else:
    model = mbnet(dataset=args.dataset)
    model.to(device)

# Use SGD optimizer
optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

# For regression, use a lower learning rate
if args.regression:
    for param_group in optimizer.param_groups:
        param_group['lr'] = args.lr * 0.1  # Use 0.01 instead of 0.1

# # additional subgradient descent on the sparsity-induced penalty term
# def updateBN():
#     if args.custom_model:
#         return
#     for m in model.modules():
#         if isinstance(m, MbBlock):
#             m.bn2.weight.grad.data.add_(args.s*torch.sign(m.bn2.weight.data))
#         elif isinstance(m, ConvBlock):
#             m.bn.weight.grad.data.add_(args.s*torch.sign(m.bn.weight.data))

def train(epoch):
    model.train()
    avg_loss = 0.
    train_acc = 0.
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        data, target = Variable(data), Variable(target)
        
        optimizer.zero_grad()
        output = model(data)
        loss = F.mse_loss(output, target, reduction='none') # Use 'none' to get per-sample loss
        avg_loss += loss.data
        
        loss.mean().backward()
        optimizer.step()
        
    return avg_loss.mean().item() 
    
def test():
    model.eval()
    test_loss = 0
    
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)

        test_loss += F.mse_loss(output, target, reduction='none')
    
    return test_loss.mean().item()

def save_checkpoint(state, model_path):
    torch.save(state, model_path)

# Set initial value for best loss
best_loss = float('inf')  # Start with highest loss. For MSE, lower is better.
# Set initial running loss
train_loss = 0.0
test_loss = 0.0

curr_lr = args.lr
if args.regression:
    curr_lr = args.lr * 0.1  # Lower LR for regression

# TQDM
epoch_bar = tqdm(
    range(args.start_epoch, args.epochs),
    desc=f'Epochs Progress',
    position=0,
    leave=True,
)

# Az, datalogging purpose
dct = {'train_loss': [], 'test_loss': []}

for epoch in epoch_bar:
    
    # if epoch in [int(args.epochs * 0.5), int(args.epochs * 0.75)]:
    #     for param_group in optimizer.param_groups:
    #         param_group['lr'] *= 0.1
    #         curr_lr *= 0.1
    # log.info('{}, {}'.format(epoch, curr_lr))

    train_loss = train(epoch)
    test_loss = test()
    
    dct['train_loss'].append(train_loss)
    dct['test_loss'].append(test_loss)
    
    is_best = test_loss < best_loss
    best_loss = min(test_loss, best_loss)
        
    # log.info('Best: {:.6f}'.format(best_loss))
    # log.info(f"Current: {test_loss:.6f} ")
    
    torch.save({
        'epoch': epoch + 1,
        'cfg': model.cfg,
        'sr': args.sr,
        's': args.s,
        'state_dict': model.state_dict(),
        'best_loss': best_loss,
        'optimizer': optimizer.state_dict(),
    }, os.path.join(args.save, model_save_path))
    
    # update tqdm
    epoch_bar.set_description(
        f'Epoch {epoch+1}/{args.epochs} | '
        f'Best Loss: {best_loss:.6f} | '
        f'Train: {train_loss:.6f} | '
        f'Test: {test_loss:.6f}'
    )
    
df = pd.DataFrame(dct)
df.to_csv(os.path.join(args.save, f'{args.dataset}_loss_log.csv'), index=False)