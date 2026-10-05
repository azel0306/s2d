import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader, TensorDataset , random_split

def get_dataloader(dataset_name, 
                    train_size = 256, 
                    test_size = 64, 
                    train_batch = 32, 
                    test_batch = 64,
                    normalize=False, standardize=False):
    """
    Create regression dataset with normalization and standardization.
    """
    
    if dataset_name == 'rastrigin':
        df = pd.read_csv('data/rastrigin_1d_pi.csv', delimiter=',')
            
        # Convert to float32 explicitly
        x = torch.tensor(df['x'].to_numpy(), dtype=torch.float32)
        y = torch.tensor(df['y'].values, dtype=torch.float32)
        
        x = x.view(-1, 1)  # Ensure x is of shape (num_samples, input_dim)
        y = y.view(-1, 1)  # Ensure y is of shape (num_samples, output_dim)
        
        dataset = TensorDataset(x, y)
        train_dataset, test_dataset = random_split(dataset, [train_size, test_size], generator=torch.Generator().manual_seed(20260908))
        print(f"Train size: {train_size}, Test size: {test_size}")

        train_loader = DataLoader(train_dataset, batch_size=train_batch, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=test_batch, shuffle=False)  
        print(f"Train batch size (total batches): {train_loader.batch_size}({len(train_loader)})  , \
              Test batch size: {test_loader.batch_size}({len(test_loader)})")

    elif dataset_name == 'rosenbrock':
        NotImplementedError("Rosenbrock dataset loading is not implemented yet.")
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
        
    return train_loader, test_loader

if __name__ == "__main__":
    train_loader, test_loader = get_dataloader('rastrigin')
    
    for idx, (data, target) in enumerate(train_loader):
        print(f"Batch {idx}:")
        print(f"Data: {data.shape}, sample: {data[0] , type(data[0])}, target: {target[0] , type(target[0])}")
        print(f"Target: {target.shape}, sample: {target[0] , type(target[0])}")
        
    