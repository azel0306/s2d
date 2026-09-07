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

import sys
from model_custom import SimpleModel
from model_custom import SpLinearBlock
from compute_flops import print_model_param_nums, print_model_param_flops

# Training settings
parser = argparse.ArgumentParser(description='PyTorch CIFAR training')
parser.add_argument('--dataset', type=str, default='cifar10',
                    help='training dataset (default: cifar100)')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=64, metavar='N',
                    help='input batch size for testing (default: 64)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                    help='learning rate (default: 0.001)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.9)')
parser.add_argument('--load', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)', required=True)
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--split-index', default="1", type=str,
                    help='#number of split', required=True)
parser.add_argument('--energy', action='store_true', default=False,
                    help='energy aware splitting')
parser.add_argument('--params', action='store_true', default=False,
                    help='paramter aware splitting')
parser.add_argument('--save', default='split/saved_models', type=str,
                    help='energy aware splitting')
parser.add_argument('--grow', type=float, default=0.3,
                    help='globally split grow rate (default: 0.2)')
parser.add_argument('--exp-name', type=str, default=None,
                    help='exp name', required=True)
parser.add_argument('--start-from-retrain', action='store_true', default=False,
                    help='retrain before splitting')
parser.add_argument('--rd', type=int, default=0,
                    help='split round')

### Az - Regression flag
parser.add_argument('--regression', action='store_true', default=False,
                    help='use regression instead of classification')
parser.add_argument('--input-dim', type=int, default=1,
                    help='input dimension for regression')
parser.add_argument('--output-dim', type=int, default=1,
                    help='output dimension for regression')
parser.add_argument('--hidden-dim', type=int, default=32,
                    help='hidden dimension for regression model')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

device = torch.device('cuda') if args.cuda else torch.device('cpu')

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# Get data loaders
if args.regression:
    try:
        from regression_dataloader import get_dataloader
        train_loader, test_loader = get_dataloader(
            dataset=args.dataset,
            train_batch_size=args.batch_size,
            test_batch_size=args.test_batch_size,
            use_cuda=args.cuda,
            input_dim=args.input_dim,
            output_dim=args.output_dim,
            normalize=False,
            standardize=True
        )
    except ImportError:
        NotImplementedError("Regression dataloader not found. Please ensure regression_dataloader.py is present.")
else:
    from dataloader import get_dataloader
    train_loader, test_loader = get_dataloader(
        dataset=args.dataset,
        train_batch_size=args.batch_size,
        test_batch_size=args.test_batch_size,
        use_cuda=args.cuda
    )

args.save = os.path.join(args.save, args.exp_name)

logging_file_path = '{}_split_{}.log'.format(args.dataset, args.split_index)
model_save_path = '{}_{}.pth.tar'.format(args.dataset, str(args.rd))

if not os.path.exists(args.save):
    os.makedirs(args.save)

#########################################################
# create file handler which logs even debug messages
import logging
log = logging.getLogger()
log.setLevel(logging.DEBUG)

ch = logging.StreamHandler()
fh = logging.FileHandler(os.path.join(args.save, logging_file_path))

formatter = logging.Formatter('%(asctime)s - %(message)s')
ch.setFormatter(formatter)
fh.setFormatter(formatter)

log.addHandler(fh)
log.addHandler(ch)
#########################################################

assert args.load
assert os.path.isfile(args.load)
log.info("=> loading checkpoint '{}'".format(args.load))
checkpoint = torch.load(args.load, weights_only=False)

# Create model with proper cfg
if args.regression:
    if 'cfg' in checkpoint:
        cfg = checkpoint['cfg']
    else:
        cfg = [(args.input_dim, args.hidden_dim), (args.hidden_dim, args.output_dim)]
else:
    cfg = checkpoint['cfg']

model = SimpleModel(dataset=args.dataset, cfg=cfg).to(device)
model.load_state_dict(checkpoint['state_dict'], strict=False)
log.info("=> loaded checkpoint '{}' ".format(args.load))
del checkpoint

def test(model):
    model.eval()
    test_loss = 0
    correct = 0
    
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        
        if args.regression:
            test_loss += F.mse_loss(output, target, reduction='sum').item()
        else:
            test_loss += F.cross_entropy(output, target, reduction='sum').item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).cpu().numpy().sum()

    test_loss /= len(test_loader.dataset)
    
    if args.regression:
        log.info('\nTest set: Average MSE: {:.6f}\n'.format(test_loss))
        return test_loss
    else:
        log.info('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / float(len(test_loader.dataset))))
        return correct / float(len(test_loader.dataset))

log.info('acc before splitting')
test(model)

import pickle
num = str(args.rd)
base = model.cfg
cfg_mask = pickle.load(open('eigen/{}_min.pkl'.format(num), 'rb'))
min_eig_vecs = pickle.load(open('eigen/{}_minv.pkl'.format(num), 'rb'))
cfg = pickle.load(open('config/delta_{}_{}.pkl'.format(args.dataset, str(int(num) + 1)), 'rb'))
cfg = cfg.tolist()

print(f"cfg_mask: {cfg_mask}")
print(f"min_eig_vecs: {min_eig_vecs}")
print("Base cfg:", base)
print("Delta cfg:", cfg)

# Build new config - SKIP final layer for splitting
new_cfg = []
prev_out = 0

for i, (in_features, out_features) in enumerate(base): # TODO this works only for FC layer
    new_cfg.append([in_features+prev_out, out_features+cfg[i]])
    prev_out = cfg[i]
    
print(f"Base cfg: {base}")
print(f"New cfg: {new_cfg}")

##################################
##### copy weights and split #####
##################################
newmodel = SimpleModel(dataset=args.dataset, cfg=new_cfg)
newmodel.to(device)

# DEBUG: Print initial shapes
print("=" * 60)
print("INITIAL SHAPES (before copying):")
for name, param in newmodel.named_parameters():
    print(f"  {name}: {param.shape}")
print("=" * 60)

layer_id_in_cfg = 0
start_mask = np.array([])
end_mask = cfg_mask[layer_id_in_cfg]

# Iterate through hidden layers
for k, (m0, m1) in enumerate(zip(model.layers, newmodel.layers)):
    print(f"k={k}: {type(m0).__name__} -> {type(m1).__name__}")
    
    if isinstance(m0, SpLinearBlock) and isinstance(m1, SpLinearBlock):
        print(f">>> Splitting SpLinearBlock at k={k}")
        
        # Get indices for splitting
        idx0 = np.squeeze(np.asarray(start_mask)) # indices of split neurons from the previous layer
        idx1 = np.squeeze(np.asarray(end_mask)) # indices of split neurons to split in the current layer
        if idx0.size == 1:
            idx0 = np.resize(idx0, (1,))
        if idx1.size == 1:
            idx1 = np.resize(idx1, (1,))
        
        print(f"  idx0: {idx0}, idx1: {idx1}")
        
        # accomodate previous layer output due to splitting
        # TODO since our model is 1 hidden layer, we can skip this step. But for generalization, we need to handle this.
        
        # update the linear weights for the current layer
        linear_weight = m0.linear.weight.data.clone()  # old [out, in]
        if idx1.size != 0:
            linear_weight[idx1.tolist(), :] /= 2.
        # print(f" concatenate linear_weight shape: {linear_weight.shape}, linear_weight[idx1.tolist(), :].shape: {linear_weight[idx1.tolist(), :].shape}")
        # print(f" idx0 size: {idx0.size}, idx1 size: {idx1.size}")
        w1 = torch.cat((linear_weight, linear_weight[idx1.tolist(), :]), 0)
        eig_v = min_eig_vecs[layer_id_in_cfg].astype(float)
        eig_v = torch.from_numpy(eig_v).float().to(device)
        if idx1.size != 0:
            eig_v[idx1.tolist(), :] /= 2.
        eig_v = torch.cat((eig_v, eig_v[idx1.tolist(), :]), 0)
        w1[idx1.tolist(), :] += 1e-2 * eig_v[idx1.tolist(), :]
        # print(f"  w1 shape: {w1.shape}, eig_v shape: {eig_v.shape}")
        # print(f"  linear_weight shape: {linear_weight.shape}, idx0 size: {idx0.size}, idx1 size: {idx1.size}")
        # print(f"  w1[linear_weight.size(0):, :].shape: {w1[linear_weight.size(0):, :].shape}, eig_v[idx1.tolist(), :].shape: {eig_v[idx1.tolist(), :].shape}")
        w1[linear_weight.size(0):, :] -= 1e-2 * eig_v[idx1.tolist(), :]
        m1.linear.weight.data = w1.clone()
        m1.linear.bias.data = torch.cat((m0.linear.bias.data.clone(), m0.linear.bias.data.clone()[idx1.tolist()]), 0)
        
        # Move to next layer in config
        layer_id_in_cfg += 1
        start_mask = end_mask
        if layer_id_in_cfg < len(cfg_mask):
            end_mask = cfg_mask[layer_id_in_cfg]
        print(f"  start_mask: {start_mask}, end_mask: {end_mask if layer_id_in_cfg < len(cfg_mask) else 'end'}")
        
# Handle final linear layer
print(f">>> Splitting final Linear at k={k}")
print(f"  Old weight: {model.linear.weight.shape}, New weight: {newmodel.linear.weight.shape}")
print(f"  start_mask: {start_mask}")

idx0 = np.squeeze(np.asarray(start_mask))
print(f"  idx0: {idx0}")

if idx0.size == 0:
    # No splitting needed
    print(f"  No splitting needed, just copying weights and bias")
    newmodel.linear.weight.data = model.linear.weight.data.clone()
    newmodel.linear.bias.data = model.linear.bias.data.clone()
    print(f"  No splitting needed, just copied")
else:
    if idx0.size == 1:
        idx0 = np.resize(idx0, (1,))
    
    print(f"fc_weight shape: {model.linear.weight.shape}, fc_bias shape: {model.linear.bias.shape}")
    fc_weight = model.linear.weight.data.clone()
    fc_weight[:, idx0.tolist()] /= 2.
    fc_bias = model.linear.bias.data.clone()

    # Duplicate the input connections
    dup_weight = fc_weight[:, idx0.tolist()].clone()
    newmodel.linear.weight.data = torch.cat((fc_weight, dup_weight), 1)
    newmodel.linear.bias.data = fc_bias.clone()
    
    print(f"  New weight shape: {newmodel.linear.weight.shape}")

# VERIFICATION: Print final shapes
print("=" * 60)
print("FINAL SHAPES (after copying):")
for name, param in newmodel.named_parameters():
    print(f"  {name}: {param.shape}")
print("=" * 60)

# Print layer info
log.info("Model cfg: {}".format(model.cfg))
log.info("New model cfg: {}".format(newmodel.cfg))

if args.regression:
    print('MSE before splitting:')
else:
    print('Accuracy before splitting:')
test(model)

if args.regression:
    print('MSE after splitting:')
else:
    print('Accuracy after splitting:')
test(newmodel)

# Save the split model
torch.save({
    'cfg': newmodel.cfg,
    'split_index': args.split_index,
    'state_dict': newmodel.state_dict(),
    'args': args,
}, os.path.join(args.save, model_save_path))

print(os.path.join(args.save, model_save_path))