from __future__ import print_function
import os
import argparse
import shutil
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import pickle
import sys
from numpy import linalg as LA
from dataloader import get_data_loader
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

# Get data loaders
if args.regression:
    # Custom regression data loader
    try:
        from regression_dataloader import get_regression_data_loader
        train_loader, test_loader = get_regression_data_loader(
            dataset=args.dataset,
            train_batch_size=args.batch_size,
            test_batch_size=args.test_batch_size,
            use_cuda=args.cuda,
            input_dim=args.input_dim,
            output_dim=args.output_dim
        )
    except ImportError:
        # Fallback to regular dataloader
        print("Warning: regression_dataloader not found, using regular dataloader")
        train_loader, test_loader = get_data_loader(
            dataset=args.dataset,
            train_batch_size=args.batch_size,
            test_batch_size=args.test_batch_size,
            use_cuda=args.cuda
        )
else:
    train_loader, test_loader = get_data_loader(
        dataset=args.dataset,
        train_batch_size=args.batch_size,
        test_batch_size=args.test_batch_size,
        use_cuda=args.cuda
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

# # DEBUG: Check which layers have dummy enabled
# print("=" * 50)
# print(f"dummy_layer parameter: {args.layer}")
# for i, layer in enumerate(model.layers):
#     if hasattr(layer, 'apply_dummy'):
#         print(f"Layer {i}: apply_dummy = {layer.apply_dummy}, has dummy: {hasattr(layer, 'dummy')}")
#         if hasattr(layer, 'dummy'):
#             print(f"  dummy length: {len(layer.dummy)}")
# print("=" * 50)

if not args.retrain:
    model.load_state_dict(checkpoint['state_dict'], strict=False)

if args.cuda: 
    model.cuda()

# Use SGD optimizer (same as author)
optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

def train(epoch):
    model.train()
    avg_loss = 0.
    train_acc = 0.
    for batch_idx, (data, target) in enumerate(train_loader):
        if args.warm == 1:
            if epoch == 0:
                for params_group in optimizer.param_groups:
                    params_group['lr'] = args.lr * (batch_idx / 23)
            if epoch == 1:
                for params_group in optimizer.param_groups:
                    params_group['lr'] = args.lr

        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        
        # MINIMAL CHANGE: Use MSE loss for regression, cross-entropy for classification
        if args.regression:
            loss = F.mse_loss(output, target)
        else:
            loss = F.cross_entropy(output, target)
            
        avg_loss += loss.data
        
        # MINIMAL CHANGE: For regression, skip accuracy calculation
        if not args.regression:
            pred = output.data.max(1, keepdim=True)[1]
            train_acc += pred.eq(target.data.view_as(pred)).cpu().numpy().sum()
            
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            log.info('Train Epoch: {} [{}/{} ({:.1f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data))

def compute_A(epoch):
    model.eval()
    avg_loss = 0.
    train_acc = 0.
    count = 0
    A = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        
        layer = args.layer
        
        # # DEBUG 1: Check dummy before forward
        # print(f"\n{'='*50}")
        # print(f"Batch {batch_idx}")
        # print(f"Layer {layer} dummy before sp_forward: {len(model.layers[layer].dummy)}")
        
        # Call sp_forward
        output = model.sp_forward(data)
        
        # # DEBUG 2: Check dummy after forward
        # print(f"Layer {layer} dummy after sp_forward: {len(model.layers[layer].dummy)}")
        # if len(model.layers[layer].dummy) > 0:
        #     print(f"  dummy[0] type: {type(model.layers[layer].dummy[0])}")
        #     print(f"  dummy[0] requires_grad: {model.layers[layer].dummy[0].requires_grad}")
        
        # MINIMAL CHANGE: Use MSE for regression, cross-entropy for classification
        if args.regression:
            loss = F.mse_loss(output, target)
        else:
            loss = F.cross_entropy(output, target)
        
        # DEBUG 3: Check before backward
        print(f"Loss value: {loss.item()}")
        print(f"Loss requires_grad: {loss.requires_grad}")
        
        loss.backward()
        
        # # DEBUG 4: Check gradients after backward
        # print(f"After backward:")
        # for i, item in enumerate(model.layers[layer].dummy):
        #     if item.grad is not None:
        #         print(f"  dummy {i} grad shape: {item.grad.shape}")
        #         print(f"  dummy {i} grad mean: {item.grad.mean().item()}")
        #     else:
        #         print(f"  dummy {i} grad is None!")
        
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
        
        # Only process first 2 batches for debugging
        if batch_idx >= 1:
            break
    
    print(f"\n{'='*50}")
    print(f"Final A shape: {A.shape if A.shape != (0,) else 'empty'}")
    
    if A.shape == (0,):
        print("ERROR: A is empty! Cannot calculate eigen.")
        return
    
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
    correct = 0
    
    for data, target in test_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        output = model(data)
        
        # MINIMAL CHANGE: Use MSE for regression, cross-entropy for classification
        if args.regression:
            test_loss += F.mse_loss(output, target, size_average=False).data
        else:
            test_loss += F.cross_entropy(output, target, size_average=False).data
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).cpu().numpy().sum()

    test_loss /= len(test_loader.dataset)
    
    # MINIMAL CHANGE: Different logging for regression vs classification
    if args.regression:
        log.info('\nTest set: Average MSE: {:.6f}\n'.format(test_loss))
        return test_loss  # Return loss for regression
    else:
        log.info('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))
        return correct / float(len(test_loader.dataset))  # Return accuracy for classification


# MINIMAL CHANGE: Initialize prec1 before loop
prec1 = None
best_prec1 = float('inf') if args.regression else 0.

for epoch in range(args.epochs):
    if epoch in [int(args.epochs * 0.5), int(args.epochs * 0.75)]:
        for param_group in optimizer.param_groups:
            param_group['lr'] *= 0.1
    
    if args.layer != -1:
        compute_A(epoch)
        # When computing A, we don't have a prec1 value
        prec1 = None
        break
    
    train(epoch)
    prec1 = test()
    
    # MINIMAL CHANGE: Different best metric logic
    if args.regression:
        is_best = prec1 < best_prec1  # Lower is better for regression
        best_prec1 = min(prec1, best_prec1)
    else:
        is_best = prec1 > best_prec1  # Higher is better for classification
        best_prec1 = max(prec1, best_prec1)

    torch.save({
        'epoch': epoch + 1,
        'state_dict': model.state_dict(),
        'cfg': model.cfg,
        'optimizer': optimizer.state_dict(),
        'acc': prec1,
    }, model_save_path)

# Only save results if we actually trained (not just computed A)
if prec1 is not None:
    print("Best metric: " + str(best_prec1))
    if not os.path.exists('result'):
        os.makedirs('result')
    pickle.dump([prec1, best_prec1], open('result/{}_{}_result.pkl'.format(args.dataset, str(args.rd)), 'wb'))
else:
    print("Skipping result save (only computed A for layer {})".format(args.layer))