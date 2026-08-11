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
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

device = torch.device('cuda') if args.cuda else torch.device('cpu')

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

train_loader, test_loader = \
    get_data_loader(dataset=args.dataset, train_batch_size=args.batch_size, 
                   test_batch_size=args.test_batch_size, use_cuda=args.cuda)

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
model = SimpleModel(dataset=args.dataset, cfg=checkpoint['cfg']).to(device)

from collections import OrderedDict
new_state_dict = OrderedDict()

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
        test_loss += F.cross_entropy(output, target, size_average=False).data
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().numpy().sum()

    test_loss /= len(test_loader.dataset)
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
cfg = np.reshape(cfg, (1, -1))
cfg = cfg.tolist()

print("Base cfg:", base)
print("Delta cfg:", cfg)

# For FC layers, cfg is a list of (in_features, out_features)
# We need to add the delta to the out_features of each layer
new_cfg = []
for i in range(len(base)):
    if isinstance(base[i], (tuple, list)):
        # For tuple (in_features, out_features)
        new_cfg.append((base[i][0], base[i][1] + len(cfg[i])))
    else:
        # For simple int (shouldn't happen for FC)
        new_cfg.append(base[i] + cfg[i])
print("New cfg:", new_cfg)

##################################
##### copy weights and split #####
##################################
newmodel = SimpleModel(dataset=args.dataset, cfg=new_cfg)
newmodel.to(device)

layer_id_in_cfg = 0
start_mask = np.array([])
end_mask = cfg_mask[layer_id_in_cfg]


print("#"*4, "Splitting", "#"*4)
# Iterate through all modules
for k, (m0, m1) in enumerate(zip(model.modules(), newmodel.modules())):
    # Check if it's a SpLinearBlock (our FC layer)
    if isinstance(m0, SpLinearBlock):
        idx0 = np.squeeze(np.argwhere(np.asarray(start_mask)))
        idx1 = np.squeeze(np.argwhere(np.asarray(end_mask)))
        
        if idx0.size == 1:
            idx0 = np.resize(idx0, (1,))
        if idx1.size == 1:
            idx1 = np.resize(idx1, (1,))
        
        # Copy linear layer weights
        # m0 is the old layer, m1 is the new layer
        linear_weight = m0.linear.weight.data.clone()  # [out_features, in_features]
        linear_bias = m0.linear.bias.data.clone()      # [out_features]
        
        # For FC: split the output neurons
        # New weight: [out_features + delta, in_features]
        # We duplicate the selected neurons and perturb them
        
        # Copy original weights
        new_weight = linear_weight.clone()
        
        # Add duplicated neurons
        if idx0.size != 0:
            # Duplicate the selected neurons
            dup_weight = linear_weight[idx0.tolist(), :].clone()
            # Divide by 2 (match author's style)
            new_weight[idx0.tolist(), :] /= 2.
            dup_weight /= 2.
            # Concatenate: original + duplicates
            new_weight = torch.cat((new_weight, dup_weight), 0)
        else:
            # If no neurons to split, just keep original
            new_weight = linear_weight.clone()
        
        # Apply eigenvector perturbation (if we have eigenvectors)
        if idx0.size != 0 and idx1.size != 0:
            eig_v = min_eig_vecs[layer_id_in_cfg].astype(float)
            eig_v = torch.from_numpy(eig_v).float().cuda()
            
            # For FC: eig_v shape is [out_features, in_features]
            # Perturb the selected neurons
            if len(eig_v.shape) == 2:
                # Split the eigenvector
                eig_v_original = eig_v[idx1.tolist(), :].clone()
                eig_v_dup = eig_v[idx1.tolist(), :].clone()
                
                # Add perturbation
                new_weight[idx1.tolist(), :] += 1e-2 * eig_v_original
                new_weight[linear_weight.size(0):, :] -= 1e-2 * eig_v_dup
            else:
                # If eig_v has different shape, handle gracefully
                print(f"Warning: Unexpected eig_v shape: {eig_v.shape}")
        
        # Assign new weights
        m1.linear.weight.data = new_weight.clone()
        
        # Copy bias (no splitting needed for bias)
        m1.linear.bias.data = linear_bias.clone()
        
        # Copy activation and dummy settings
        m1.activation = m0.activation
        m1.apply_dummy = m0.apply_dummy
        
        # Move to next layer in config
        layer_id_in_cfg += 1
        start_mask = end_mask
        if layer_id_in_cfg < len(cfg_mask):
            end_mask = cfg_mask[layer_id_in_cfg]
        print("start_mask:", start_mask)
            
    # Handle final linear layer (if not already handled)
    elif isinstance(m0, nn.Linear) and not isinstance(m0, SpLinearBlock):
        print("#"*4, "Final Linear Layer", "#"*4)
        idx0 = np.squeeze(np.asarray(start_mask))
        print("idx0:", idx0)
        #if idx0.size == 0: continue
        if idx0.size == 1:
            idx0 = np.resize(idx0, (1,))
        fc_weight = m0.weight.data.clone()
        fc_weight[:, idx0.tolist()] /= 2.
        fc_bias = m0.bias.data.clone()
        m1.weight.data = torch.cat((fc_weight, fc_weight[:, idx0.tolist()]), 1)
        m1.bias.data = m0.bias.data.clone()
    
# Print layer info
log.info("Model cfg: {}".format(model.cfg))
log.info("New model cfg: {}".format(newmodel.cfg))
print('acc after splitting')
test(newmodel)
test(model)

# Save the split model
torch.save({
    'cfg': newmodel.cfg,
    'split_index': args.split_index,
    'state_dict': newmodel.state_dict(),
    'args': args,
}, os.path.join(args.save, model_save_path))

print(os.path.join(args.save, model_save_path))