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
parser.add_argument('--total-layers', type=int, default=3, help="number of hidden layers to create")

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device('cuda') if args.cuda else torch.device('cpu')

print('\n' + '=' * 70)
print('TRAINING CONFIGURATION')
print('=' * 70)
for key, value in vars(args).items():
    label = key.replace('_', ' ').title()
    print(f'{label:<14}: {value}')
print('=' * 70 + '\n')


torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
    
if args.custom_model:
    from model_custom import SimpleModel 
    
    logging_file = f'{args.dataset}.log'
    model_filename = f'{args.dataset}_0.pth.tar'
    subdir1 = f'{args.dataset}_hid{args.total_layers}_{args.hidden_dim}neuron'
    subdir2 = f'run_{args.seed}'

    model_dir = os.path.join(args.save, subdir1, subdir2)

    if not os.path.exists(model_dir):
        print(f'creating subfolder at {model_dir}')
        os.makedirs(model_dir)

    print(f"Model will be saved to... {model_dir}")

# else:
#     if args.sp:
#         from sp_mbnet import sp_mbnet as mbnet
#         from sp_mbnet import splitcfg
#     else:
#         from mobilenetv1 import MobileNetV1 as mbnet
#         from mobilenetv1 import MbBlock, ConvBlock

#     assert not (args.sr and args.sp)
    
#     logging_file = '{}.log'.format(args.dataset)
#     model_save_path = '{}_0.pth.tar'.format(args.dataset)

#     if not os.path.exists(args.save):
#         os.makedirs(args.save)

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
            train_batch=args.batch_size,
            test_batch=args.test_batch_size,
            normalize=False,      
            standardize=False      
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
        input_dim = args.input_dim
        hidden_dim = args.hidden_dim
        output_dim = args.output_dim
        total_layers = args.total_layers
        first_layer = (input_dim, hidden_dim)
        subsequent_layers = [(hidden_dim, hidden_dim) for idx in range(total_layers-1)]
        cfg = [first_layer, *subsequent_layers]
        print(f"Model config: {cfg}")
    else:
        cfg = None
    model = SimpleModel(cfg=cfg, dataset=args.dataset, activation='relu')
    print("Model details:", model)
    model.to(device)
else:
    model = mbnet(dataset=args.dataset)
    model.to(device)

# Use SGD optimizer
optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

def train(epoch):
    model.train()
    avg_loss = []
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        data, target = Variable(data), Variable(target)
        
        optimizer.zero_grad()
        output = model(data)
        loss = F.mse_loss(output, target, reduction='none') # Use 'none' to get per-sample loss
        avg_loss.extend(loss.data.view(-1,).cpu().numpy())
        
        loss.mean().backward()
        optimizer.step()
        
    return np.array(avg_loss).mean()
    
def test():
    model.eval()
    avg_loss = []
    
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)

        loss = F.mse_loss(output, target, reduction='none')
        avg_loss.extend(loss.data.view(-1,).cpu().numpy())
    
    return np.array(avg_loss).mean()

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
    train_loss = train(epoch)
    test_loss = test()
    
    dct['train_loss'].append(train_loss)
    dct['test_loss'].append(test_loss)
    
    is_best = test_loss < best_loss
    best_loss = min(test_loss, best_loss)
    
    torch.save({
        'epoch': epoch + 1,
        'cfg': model.cfg,
        'sr': args.sr,
        's': args.s,
        'state_dict': model.state_dict(),
        'best_loss': best_loss,
        'optimizer': optimizer.state_dict(),
    }, os.path.join(model_dir, model_filename))
    
    # update tqdm
    epoch_bar.set_description(
        f'Epoch {epoch+1}/{args.epochs} | '
        f'Best Loss: {best_loss:.6f} | '
        f'Train: {train_loss:.6f} | '
        f'Test: {test_loss:.6f}'
    )
    
df = pd.DataFrame(dct)
df.to_csv(os.path.join(model_dir, f'{args.dataset}_loss_log_0.csv'), index=False)