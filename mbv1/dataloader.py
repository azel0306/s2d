from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
import os


#dir_path = os.path.dirname(os.path.realpath(__file__))
base_data_dir = './'
def get_data_loader(dataset='mnist', train_batch_size=100, test_batch_size=100, use_cuda=True):

    kwargs = {'num_workers': 8, 'pin_memory': True} if use_cuda else {}
    if dataset == 'mnist':
        train_loader = torch.utils.data.DataLoader(
            datasets.MNIST(os.path.join(base_data_dir, './data_mnist'), train=True, download=True,
                           transform=transforms.Compose([
                               transforms.Resize(32),
                               transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,))
                           ])),
            batch_size=train_batch_size, shuffle=True, **kwargs)
        test_loader = torch.utils.data.DataLoader(
            datasets.MNIST(os.path.join(base_data_dir, './data_mnist'), train=False, transform=transforms.Compose([
                               transforms.Resize(32),
                               transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,))
                           ])),
            batch_size=test_batch_size, shuffle=True, **kwargs)

    elif dataset == 'fashion_mnist':
        train_loader = torch.utils.data.DataLoader(
            datasets.FashionMNIST(os.path.join(base_data_dir, './data_fashion_mnist'), train=True, download=True,
                           transform=transforms.Compose([
                               transforms.Resize(32),
                               transforms.ToTensor(),
                           ])),
            batch_size=train_batch_size, shuffle=True, **kwargs)
        test_loader = torch.utils.data.DataLoader(
            datasets.MNIST(os.path.join(base_data_dir, './data_fashion_mnist'), train=False, transform=transforms.Compose([
                               transforms.Resize(32),
                               transforms.ToTensor(),
                           ])),
            batch_size=test_batch_size, shuffle=True, **kwargs)

    elif dataset == 'cifar10':
        train_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10(os.path.join(base_data_dir, './data.cifar10'), train=True, download=True,
                           transform=transforms.Compose([
                               transforms.Pad(4),
                               transforms.RandomCrop(32),
                               transforms.RandomHorizontalFlip(),
                               transforms.ToTensor(),
                               transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                           ])),
            batch_size=train_batch_size, shuffle=True, **kwargs)
        test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10(os.path.join(base_data_dir, './data.cifar10'), train=False, transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                           ])),
            batch_size=test_batch_size, shuffle=True, **kwargs)

    elif dataset == 'cifar100':
        train_loader = torch.utils.data.DataLoader(
            datasets.CIFAR100(os.path.join(base_data_dir, './data.cifar100'), train=True, download=True,
                           transform=transforms.Compose([
                               transforms.Pad(4),
                               transforms.RandomCrop(32),
                               transforms.RandomHorizontalFlip(),
                               transforms.ToTensor(),
                               transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                           ])),
            batch_size=train_batch_size, shuffle=True, **kwargs)
        test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR100(os.path.join(base_data_dir, './data.cifar100'), train=False, transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                           ])),
            batch_size=test_batch_size, shuffle=True, **kwargs)
        
    elif dataset == 'rastrin' or dataset == 'rosenbrock':
        torch.manual_seed(0)
        
        def rosenbrock(n_samples = 200+1):
            x = torch.linspace(-2, 2, n_samples)
            y = 100 * (x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2
            return x[1:], y

        def rastrin(n_samples = 200):
            x = torch.linspace(-5.12, 5.12, n_samples)
            A = 10
            y = A * n_samples + (x ** 2 - A * torch.cos(2 * np.pi * x))

            return x, y
        
        x, y = rosenbrock() if dataset == 'rosenbrock' else rastrin()
        x, y = x.unsqueeze(1), y.unsqueeze(1)
    
        # === 2. Apply normalization and standardization ===
        # Normalize x to [0, 1] range
        x_normalized, x_min, x_max = normalize(x)
        # Standardize y to zero mean and unit variance
        y_standardized, y_mean, y_std = standardize(y)
        
        print(f"x normalized range: [{x_normalized.min():.4f}, {x_normalized.max():.4f}]")
        print(f"y standardized: mean={y_mean:.4f}, std={y_std:.4f}")

        # Use normalized/standardized data for training
        x = x_normalized.cuda() if use_cuda else x_normalized
        y = y_standardized.cuda() if use_cuda else y_standardized
        
        train_dataset = CustomDataset(x, y)
        test_dataset = CustomDataset(x, y)  # Using the same data for testing for now; in practice, you would want a separate test set
        
        train_loader = DataLoader(train_dataset, batch_size=200, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=200, shuffle=True)

    return train_loader, test_loader

class CustomDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

def normalize(data, min_val=None, max_val=None):
    """Min-max normalization to [0, 1] range"""
    if min_val is None:
        min_val = data.min()
    if max_val is None:
        max_val = data.max()
    return (data - min_val) / (max_val - min_val + 1e-8), min_val, max_val

def denormalize(data, min_val, max_val):
    """Reverse min-max normalization"""
    return data * (max_val - min_val + 1e-8) + min_val

def standardize(data, mean=None, std=None):
    """Standardization to zero mean and unit variance"""
    if mean is None:
        mean = data.mean()
    if std is None:
        std = data.std()
    return (data - mean) / (std + 1e-8), mean, std

def destandardize(data, mean, std):
    """Reverse standardization"""
    return data * (std + 1e-8) + mean

if __name__ == "__main__":
    train_loader, test_loader = get_data_loader(dataset='rosenbrock', train_batch_size=10, test_batch_size=10, use_cuda=False)
    
    for idx, (data, target) in enumerate(train_loader):
        print(f"Batch {idx}:")
        print(f"Data: {data}")
        print(f"Target: {target}")