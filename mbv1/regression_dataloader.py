import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader, TensorDataset

def get_dataloader(dataset='rosenbrock', train_batch_size=200, 
                               test_batch_size=200, use_cuda=True, 
                               input_dim=1, output_dim=1, num_samples=200,
                               normalize=True, standardize=True):
    """
    Create regression dataset with normalization and standardization.
    """
    
    if dataset == 'rastrin' or dataset == 'rosenbrock':
        if dataset == 'rastrin':
            df = pd.read_csv('data/rastrin.csv', delimiter=',')
        else:
            df = pd.read_csv('data/rosenbrock.csv', delimiter=',')
            
        # Convert to float32 explicitly
        x = torch.tensor(df['x'].to_numpy(), dtype=torch.float32)
        y_true = torch.tensor(df['y'].values, dtype=torch.float32)
        y_train = torch.tensor(df['y_train'].values, dtype=torch.float32)
        y_test = torch.tensor(df['y_test'].values, dtype=torch.float32)      
        
        # if normalize:
        #     x = normalize_data(x)
        #     y_true, min, max = normalize_data(y_true)
        #     y_train, _, _ = normalize_data(y_train, min_val=min, max_val=max)
        #     y_test, _, _ = normalize_data(y_test, min_val=min, max_val=max)
        
        # if standardize:
        #     x = standardize_data(x)
        #     y_true, mean, std = standardize_data(y_true)
        #     y_train, _, _ = standardize_data(y_train, mean_val=mean, std_val=std)
        #     y_test, _, _ = standardize_data(y_test, mean_val=mean, std_val=std)
        
        # === 2. Apply normalization and standardization ===
        # Normalize x to [0, 1] range
        x_normalized, x_min, x_max = normalize_data(x)
        # Standardize y to zero mean and unit variance
        y_true_std, y_true_mean, y_true_std = standardize_data(y_true)
        y_train_data, y_train_mean, y_train_std = standardize_data(y_train)
        y_test_data, y_test_mean, y_test_std = standardize_data(y_test)
        
        x_normalized = x_normalized.unsqueeze(1)  # Add feature dimension
        y_train_data = y_train_data.unsqueeze(1)  # Add feature dimension
        y_test_data = y_test_data.unsqueeze(1)    # Add feature dimension
        
        print(x_normalized.shape, y_train_mean.shape)
        print(f"x normalized range: [{x_normalized.min():.4f}, {x_normalized.max():.4f}]")
        print(f"y standardized: mean={y_train_mean:.4f}, std={y_train_std:.4f}")
        
        
        train_data = TensorDataset(x_normalized, y_train_data)
        test_data = TensorDataset(x_normalized, y_test_data)
        train_loader = DataLoader(train_data, batch_size=train_batch_size, shuffle=False)
        # TODO think of a way for the test loader, temporarily using training data for testing
        test_loader = DataLoader(test_data, batch_size=test_batch_size, shuffle=False)  
        
        return train_loader, test_loader
    

    return train_loader, test_loader

    
class CustomDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

def normalize_data(data, min_val=None, max_val=None):
    """Min-max normalization to [0, 1] range"""
    if min_val is None:
        min_val = data.min()
    if max_val is None:
        max_val = data.max()
    return (data - min_val) / (max_val - min_val + 1e-8), min_val, max_val

def denormalize_data(data, min_val, max_val):
    """Reverse min-max normalization"""
    return data * (max_val - min_val + 1e-8) + min_val

def standardize_data(data, mean=None, std=None):
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
    train_loader, test_loader = get_dataloader(dataset='rosenbrock', train_batch_size=200, test_batch_size=200, use_cuda=False)
    
    for idx, (data, target) in enumerate(train_loader):
        print(f"Batch {idx}:")
        print(f"Data: {data.shape}, sample: {data[0] , type(data[0])}")
        print(f"Target: {target.shape}, sample: {target[0] , type(target[0])}")