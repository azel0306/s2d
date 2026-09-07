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
from dataloader import get_data_loader
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
        from regression_dataloader import get_regression_data_loader
        train_loader, test_loader = get_regression_data_loader(
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
prev_out = None

for i in range(len(base)):
    if isinstance(base[i], (tuple, list)):
        in_features, out_features = base[i]
        
        if prev_out is not None:
            in_features = prev_out
        
        # ONLY add delta if this is NOT the final layer
        if i < len(base) - 1:
            new_out = out_features + cfg[i]
        else:
            # Final layer: keep output dimension the same
            new_out = out_features
        
        new_cfg.append((in_features, new_out))
        prev_out = new_out
    else:
        if i < len(base) - 1:
            new_cfg.append(base[i] + cfg[i])
        else:
            new_cfg.append(base[i])
        prev_out = base[i] + cfg[i] if i < len(base) - 1 else base[i]

print("New cfg:", new_cfg)

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

# Iterate through modules
for k, (m0, m1) in enumerate(zip(model.modules(), newmodel.modules())):
    print(f"k={k}: {type(m0).__name__} -> {type(m1).__name__}")
    
    # Check if it's a SpLinearBlock
    if isinstance(m0, SpLinearBlock) and isinstance(m1, SpLinearBlock):
        print(f">>> Splitting SpLinearBlock at k={k}")
        print(f"  Old weight: {m0.linear.weight.shape}, New weight: {m1.linear.weight.shape}")
        
        # Get indices for splitting
        idx0 = np.squeeze(np.argwhere(np.asarray(start_mask)))
        idx1 = np.squeeze(np.argwhere(np.asarray(end_mask)))
        
        print(f"  idx0: {idx0}, idx1: {idx1}")
        
        if idx0.size == 0:
            # No neurons to split - just copy
            m1.linear.weight.data = m0.linear.weight.data.clone()
            m1.linear.bias.data = m0.linear.bias.data.clone()
            print(f"  No neurons to split, just copied")
        else:
            if idx0.size == 1:
                idx0 = np.resize(idx0, (1,))
            if idx1.size == 1:
                idx1 = np.resize(idx1, (1,))
            
            # Copy and split weights
            linear_weight = m0.linear.weight.data.clone()  # [old_out, in]
            linear_bias = m0.linear.bias.data.clone()      # [old_out]
            
            # Duplicate selected neurons
            dup_weight = linear_weight[idx0.tolist(), :].clone()
            
            # Divide original and duplicates by 2
            new_weight = linear_weight.clone()
            new_weight[idx0.tolist(), :] /= 2.
            dup_weight /= 2.
            
            # Concatenate: original + duplicates
            new_weight = torch.cat((new_weight, dup_weight), 0)
            
            # Apply eigenvector perturbation
            if idx1.size != 0 and layer_id_in_cfg in min_eig_vecs:
                eig_v = min_eig_vecs[layer_id_in_cfg].astype(float)
                eig_v = torch.from_numpy(eig_v).float().to(device)
                
                if len(eig_v.shape) == 2:
                    # Perturb the split neurons
                    new_weight[idx1.tolist(), :] += 1e-2 * eig_v[idx1.tolist(), :]
                    new_weight[linear_weight.size(0):, :] -= 1e-2 * eig_v[idx1.tolist(), :]
                    print(f"  Applied eigenvector perturbation")
                else:
                    print(f"  Warning: Unexpected eig_v shape: {eig_v.shape}")
            
            # Assign to new model
            m1.linear.weight.data = new_weight.clone()
            m1.linear.bias.data = linear_bias.clone()
            
            print(f"  New weight shape: {m1.linear.weight.shape}")
        
        # Copy activation and dummy settings
        m1.activation = m0.activation
        m1.apply_dummy = m0.apply_dummy
        
        # Move to next layer in config
        layer_id_in_cfg += 1
        start_mask = end_mask
        if layer_id_in_cfg < len(cfg_mask):
            end_mask = cfg_mask[layer_id_in_cfg]
        print(f"  start_mask: {start_mask}, end_mask: {end_mask if layer_id_in_cfg < len(cfg_mask) else 'end'}")
            
    # Handle final linear layer
    elif isinstance(m0, nn.Linear) and not isinstance(m0, SpLinearBlock):
        print(f">>> Splitting final Linear at k={k}")
        print(f"  Old weight: {m0.weight.shape}, New weight: {m1.weight.shape}")
        print(f"  start_mask: {start_mask}")
        
        idx0 = np.squeeze(np.asarray(start_mask))
        print(f"  idx0: {idx0}")
        
        if idx0.size == 0:
            # No splitting needed
            m1.weight.data = m0.weight.data.clone()
            m1.bias.data = m0.bias.data.clone()
            print(f"  No splitting needed, just copied")
        else:
            if idx0.size == 1:
                idx0 = np.resize(idx0, (1,))
            
            print(f"fc_weight shape: {m0.weight.shape}, fc_bias shape: {m0.bias.shape}")
            fc_weight = m0.weight.data.clone()
            fc_weight[:, idx0.tolist()] /= 2.
            fc_bias = m0.bias.data.clone()
            
            # Duplicate the input connections
            dup_weight = fc_weight[:, idx0.tolist()].clone()
            m1.weight.data = torch.cat((fc_weight, dup_weight), 1)
            m1.bias.data = fc_bias.clone()
            
            print(f"  New weight shape: {m1.weight.shape}")

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