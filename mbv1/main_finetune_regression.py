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
from torchvision import datasets, transforms
from torch.autograd import Variable
import pickle
import sys
from numpy import linalg as LA
from tqdm import tqdm
from compute_flops import print_model_param_nums, print_model_param_flops
import json

# Training settings
parser = argparse.ArgumentParser(description='PyTorch CIFAR training')
parser.add_argument('--dataset', type=str, default='cifar10',
                    help='training dataset (default: cifar100)', required=True)
parser.add_argument('--load', default='', type=str, metavar='PATH',
                    help='path to the pruned/split model to be fine tuned', required=True)
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 256)')
parser.add_argument('--test-batch-size', type=int, default=100, metavar='N',
                    help='input batch size for testing (default: 100)')
parser.add_argument('--epochs', type=int, default=160, metavar='N',
                    help='number of epochs to train (default: 160)')
parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                    help='learning rate (default: 0.1)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.9)')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--sp', action='store_true', default=True,
                    help='splitting settings')
parser.add_argument('--retrain', action='store_true', default=False,
                    help='retrain, otherwise, finetune')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--layer', type=int, default=-1,
                    help='layer to split')
parser.add_argument('--warm', type=int, default=0,
                    help='warm up the training')
parser.add_argument('--rd', type=int, default=0,
                    help='split round')

### Az - Custom model flag
parser.add_argument('--custom_model', action='store_true', default=False,
                    help='use custom model')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed for model initialization (default: 1)')

### Az - Regression dataset flag
parser.add_argument('--regression', action='store_true', default=False,
                    help='use regression instead of classification')

### Az - Input/output dimensions for regression
parser.add_argument('--input-dim', type=int, default=1,
                    help='input dimension for regression')
parser.add_argument('--output-dim', type=int, default=1,
                    help='output dimension for regression')
parser.add_argument('--hidden-dim', type=int, default=3,
                    help='hidden dimension for regression model')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device('cuda') if args.cuda else torch.device('cpu')

# Import model
if args.custom_model:
    from model_custom import SimpleModel as custom_model
elif args.sp:
    from sp_mbnet import sp_mbnet as mbnet
else:
    from mobilenetv1 import MobileNetV1 as mbnet

assert args.load
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

args.save = os.path.dirname(args.load)

if args.retrain:
    args.save = os.path.join(args.save, 'retrain')
else:
    args.save = os.path.join(args.save, 'finetune')

if not os.path.exists(args.save):
    os.makedirs(args.save)

print("#"*64)
print(f"Using checkpoint {args.load}")
print("#"*64)

checkpoint = torch.load(args.load, weights_only=False)

if args.custom_model and args.layer == -1:
    model_save_path = os.path.join(args.save,'{}_{}.pth.tar'.format(args.dataset, str(int(args.rd) + 1)))
elif args.sp and args.layer == -1 and not args.custom_model:
    model_save_path = os.path.join(args.save,'{}_{}.pth.tar'.format(args.dataset, str(int(args.rd) + 1)))

if args.layer == -1:
    logging_file_path = model_save_path.replace(".pth.tar", ".log")
else:
    logging_file_path = str(args.layer) + '.log'

print('====================ROUND========================', args.rd)
#########################################################
# create file handler which logs even debug messages
import logging
log = logging.getLogger()
log.setLevel(logging.DEBUG)

ch = logging.StreamHandler()
fh = logging.FileHandler(logging_file_path)

formatter = logging.Formatter('%(asctime)s - %(message)s')
ch.setFormatter(formatter)
fh.setFormatter(formatter)

log.addHandler(fh)
log.addHandler(ch)
#########################################################

from regression_dataloader import get_dataloader
train_loader, test_loader = get_dataloader(
    args.dataset,
    train_batch_size=args.batch_size,
    test_batch_size=args.test_batch_size,
    use_cuda=args.cuda,
    input_dim=args.input_dim,
    output_dim=args.output_dim
)

print("args.custom_model", args.custom_model)

# Create model
if args.custom_model:
    # Build cfg for regression
    if args.regression:
        cfg = checkpoint.get('cfg', None)
        if cfg is None:
            # If no cfg in checkpoint, create a simple cfg for regression
            print("No cfg found in checkpoint, creating default regression cfg")
            cfg = [(args.input_dim, args.hidden_dim), (args.hidden_dim, args.output_dim)]
        else:
            print("Using cfg from checkpoint:", cfg)
        print("layer", args.layer)
    else: 
    # Build cfg for classification
        NotImplementedError
    
    model = custom_model(cfg=cfg, dataset=args.dataset, activation='relu', dummy_layer=args.layer)
else:
    model = mbnet(cfg=checkpoint['cfg'], dataset=args.dataset, dummy_layer=args.layer)

# load weights, otherwise, only the arch is used
print("Model cfg:", model.cfg)
if not os.path.exists('config'):
    os.makedirs('config')
if args.layer != -1:
    pickle.dump(model.cfg, open('config/{}_{}.pkl'.format(args.dataset, str(args.rd)), 'wb'))
else:
    pickle.dump(model.cfg, open('config/{}_{}.pkl'.format(args.dataset, str(args.rd + 1)), 'wb'))

if not args.retrain:
    model.load_state_dict(checkpoint['state_dict'], strict=False)

if args.cuda: 
    model.cuda()

# Use SGD optimizer (same as author)
optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

def train(epoch):
    model.train()
    avg_loss = 0.
    
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

def compute_A(epoch):
    model.eval()
    avg_loss = 0.
    count = 0
    A = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        data, target = Variable(data), Variable(target)
        
        optimizer.zero_grad()

        layer = args.layer
        
        # Call sp_forward
        output = model.sp_forward(data)
        loss = F.mse_loss(output, target, reduction='none') # Use 'none' to get per-sample loss
        avg_loss += loss.data
        
        loss.mean().backward()
        
        # Collect gradients
        a = [item.grad.data.cpu().numpy() for item in model.layers[layer].dummy]
        print(f"Collected a length: {len(a)}")
        
        if len(a) > 0:
            print(f"a[0] shape: {a[0].shape}")
            a = np.array(a)
            print(f"a array shape: {a.shape}")
        else:
            print("WARNING: a is empty!")
            # If empty, create a dummy array to avoid crash
            a = np.array([])
        
        if count == 0:
            A = a
        else:
            if A.shape == (0,) or a.shape == (0,):
                print("WARNING: A or a is empty, skipping accumulation")
            else:
                A += a
        count += 1
        
    
    # print(f"\n{'='*50}")
    # print(f"Final A shape: {A.shape if A.shape != (0,) else 'empty'}")

    # if A.shape == (0,):
    #     print("ERROR: A is empty! Cannot calculate eigen.")
    #     return
    
    A = np.array(A)
    A = A / count
    print(f"A after averaging shape: {A.shape}")
    
    rd = args.rd
    calculate_eigen(A, args.layer, rd)


def calculate_eigen(A, layer, rd=0):
    A = (A + np.transpose(A, [0, 2, 1])) / 2
    w, v = LA.eig(A)
    w_min = np.min(w, axis=1)
    V = []
    for a in range(w_min.shape[0]):
        amina = np.argmin(w[a])
        V.append(v[a, :, amina])
    V = np.array(V)
    if not os.path.exists('eigen'):
        os.makedirs('eigen')
    pickle.dump(w_min, open('eigen/{}_A_{}_{}_.pkl'.format(args.dataset, str(layer), str(rd)), 'wb'))
    pickle.dump(V, open('eigen/{}_V_{}_{}_.pkl'.format(args.dataset, str(layer), str(rd)), 'wb'))


@torch.no_grad()
def test():
    model.eval()
    test_loss = 0
    
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        data, target = Variable(data), Variable(target)
        output = model(data)
        
        test_loss += F.mse_loss(output, target, reduction='none')
    
    return test_loss.mean().item()  # Return loss for regression


# MINIMAL CHANGE: Initialize loss before loop
train_loss = 0.0
test_loss = 0.0
best_loss = float('inf')

# TQDM
epoch_bar = tqdm(
    range(args.epochs),
    desc=f'Epochs Progress',
    position=0,
    leave=True,
)

dct = {'train_loss': [], 'test_loss': []} # data logging purpose

for epoch in range(args.epochs):
    # if epoch in [int(args.epochs * 0.5), int(args.epochs * 0.75)]:
    #     for param_group in optimizer.param_groups:
    #         param_group['lr'] *= 0.1
    
    if args.layer != -1:
        compute_A(epoch)
        # When computing A, we don't have a loss value
        loss = None
        break
    
    train_loss = train(epoch)
    test_loss = test()
    
    dct['train_loss'].append(train_loss)
    dct['test_loss'].append(test_loss)
    
    is_best = test_loss < best_loss  # Lower is better for regression
    best_loss = min(test_loss, best_loss)

    torch.save({
        'epoch': epoch + 1,
        'state_dict': model.state_dict(),
        'cfg': model.cfg,
        'optimizer': optimizer.state_dict(),
        'acc': test_loss,
    }, model_save_path)
    
    # update tqdm
    epoch_bar.set_description(
        f'Epoch {epoch+1}/{args.epochs} | '
        f'Best Loss: {best_loss:.6f} | '
        f'Train: {train_loss:.6f} | '
        f'Test: {test_loss:.6f}'
    )

# Only save results if we actually trained (not just computed A)
if test_loss is not None:
    print("Best metric: " + str(best_loss))
    if not os.path.exists('result'):
        os.makedirs('result')
    pickle.dump([test_loss, best_loss], open('result/{}_{}_result.pkl'.format(args.dataset, str(args.rd)), 'wb'))
else:
    print("Skipping result save (only computed A for layer {})".format(args.layer))

if args.layer == -1:
    print("Saving loss log to CSV")
    df = pd.DataFrame(dct)
    df.to_csv(os.path.join(args.save, f'{args.dataset}_loss_log.csv'), index=False)