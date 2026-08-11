import torch
import numpy as np

def get_regression_data_loader(dataset='rosenbrock', train_batch_size=256, 
                               test_batch_size=100, use_cuda=True, 
                               input_dim=1, output_dim=1, num_samples=10000,
                               normalize=True, standardize=True):
    """
    Create regression dataset with normalization and standardization.
    """
    kwargs = {'num_workers': 8, 'pin_memory': True} if use_cuda else {}
    
    if dataset == 'rosenbrock':
        # CORRECT Rosenbrock function for regression
        # For input_dim=1: f(x) = 100*x^4 + (1-x)^2
        # For input_dim=2: f(x,y) = (1-x)^2 + 100*(y-x^2)^2
        def rosenbrock(x):
            if x.shape[1] == 1:
                # 1D Rosenbrock: f(x) = 100*x^4 + (1-x)^2
                return 100 * x**4 + (1 - x)**2
            else:
                # 2D Rosenbrock: f(x,y) = (1-x)^2 + 100*(y-x^2)^2
                # For higher dimensions, sum over pairs
                result = torch.zeros(x.shape[0], 1)
                for i in range(0, x.shape[1] - 1, 2):
                    x_i = x[:, i:i+1]
                    x_next = x[:, i+1:i+2]
                    term = (1 - x_i)**2 + 100 * (x_next - x_i**2)**2
                    result += term
                return result
        
        # Generate training data
        x_train = torch.rand(num_samples, input_dim) * 4 - 2  # Range [-2, 2]
        y_train = rosenbrock(x_train)
        
        # Generate test data
        x_test = torch.linspace(-2, 2, 200).reshape(-1, input_dim)
        y_test = rosenbrock(x_test)
        
    elif dataset == 'rosenbrock_1d_standard':
        # Standard 1D Rosenbrock: f(x) = 100*x^4 + (1-x)^2
        def rosenbrock_1d(x):
            return 100 * x**4 + (1 - x)**2
        
        x_train = torch.rand(num_samples, input_dim) * 4 - 2
        y_train = rosenbrock_1d(x_train)
        
        x_test = torch.linspace(-2, 2, 200).reshape(-1, input_dim)
        y_test = rosenbrock_1d(x_test)
        
    elif dataset == 'rosenbrock_2d':
        # Standard 2D Rosenbrock: f(x,y) = (1-x)^2 + 100*(y-x^2)^2
        def rosenbrock_2d(x):
            # x should have 2 dimensions
            x1 = x[:, 0:1]
            x2 = x[:, 1:2]
            return (1 - x1)**2 + 100 * (x2 - x1**2)**2
        
        x_train = torch.rand(num_samples, input_dim) * 4 - 2
        y_train = rosenbrock_2d(x_train)
        
        x_test = torch.linspace(-2, 2, 200).reshape(-1, input_dim)
        y_test = rosenbrock_2d(x_test)
        
    elif dataset == 'rastrin':
        # Correct Rastrigin function
        def rastrigin(x):
            n = x.shape[1]  # Number of dimensions
            sum_term = (x**2 - 10 * torch.cos(2 * np.pi * x)).sum(dim=1, keepdim=True)
            return 10 * n + sum_term
        
        x_train = torch.rand(num_samples, input_dim) * 10.24 - 5.12
        y_train = rastrigin(x_train)
        
        x_test = torch.linspace(-5.12, 5.12, 200).reshape(-1, input_dim)
        y_test = rastrigin(x_test)
        
    elif dataset == 'sine':
        # Sine function with noise
        x_train = torch.rand(num_samples, input_dim) * 4 * np.pi - 2 * np.pi
        y_train = torch.sin(x_train) + 0.1 * torch.randn(num_samples, input_dim)
        
        x_test = torch.linspace(-2*np.pi, 2*np.pi, 200).reshape(-1, input_dim)
        y_test = torch.sin(x_test)
        
    else:
        # Default: linear function with noise
        w_true = torch.randn(input_dim, output_dim)
        b_true = torch.randn(output_dim)
        
        x_train = torch.rand(num_samples, input_dim) * 2 - 1
        y_train = x_train @ w_true + b_true + 0.1 * torch.randn(num_samples, output_dim)
        
        x_test = torch.linspace(-1, 1, 200).reshape(-1, input_dim)
        y_test = x_test @ w_true + b_true
    
    # Normalize inputs to [0, 1]
    if normalize:
        x_min = x_train.min(dim=0)[0]
        x_max = x_train.max(dim=0)[0]
        x_range = x_max - x_min
        x_range[x_range == 0] = 1.0
        
        x_train = (x_train - x_min) / x_range
        x_test = (x_test - x_min) / x_range
        
        # Also normalize outputs
        y_min = y_train.min(dim=0)[0]
        y_max = y_train.max(dim=0)[0]
        y_range = y_max - y_min
        y_range[y_range == 0] = 1.0
        
        y_train = (y_train - y_min) / y_range
        y_test = (y_test - y_min) / y_range
    
    # Standardize outputs (zero mean, unit variance)
    if standardize:
        y_mean = y_train.mean(dim=0)
        y_std = y_train.std(dim=0)
        y_std[y_std == 0] = 1.0
        
        y_train = (y_train - y_mean) / y_std
        y_test = (y_test - y_mean) / y_std
        
        # Also standardize inputs
        x_mean = x_train.mean(dim=0)
        x_std = x_train.std(dim=0)
        x_std[x_std == 0] = 1.0
        
        x_train = (x_train - x_mean) / x_std
        x_test = (x_test - x_mean) / x_std
    
    print(f"Dataset: {dataset}, input_dim: {input_dim}, output_dim: {output_dim}")
    print(f"Training samples: {num_samples}")
    print(f"x_train shape: {x_train.shape}, y_train shape: {y_train.shape}")
    print(f"x stats - mean: {x_train.mean():.4f}, std: {x_train.std():.4f}")
    print(f"y stats - mean: {y_train.mean():.4f}, std: {y_train.std():.4f}")
    
    # Create datasets
    train_dataset = torch.utils.data.TensorDataset(x_train, y_train)
    test_dataset = torch.utils.data.TensorDataset(x_test, y_test)
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=train_batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=test_batch_size, shuffle=False, **kwargs)
    
    return train_loader, test_loader


# If you want your original time-series style Rosenbrock
def generate_rosenbrock_sequence(n_samples=200):
    """
    Generate Rosenbrock as a sequence (your original implementation).
    This is for time-series prediction, NOT standard regression.
    """
    x = torch.linspace(-2, 2, n_samples)
    y = 100 * (x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2
    return x[1:], y


if __name__ == "__main__":
    print("=" * 50)
    print("Testing Rosenbrock (1D standard)")
    print("=" * 50)
    
    train_loader, test_loader = get_regression_data_loader(
        dataset='rosenbrock',
        input_dim=1,
        output_dim=1,
        num_samples=1000,
        normalize=True,
        standardize=True
    )
    
    
    for x, y in train_loader:
        print(len(train_loader))
        print(f"x shape: {x.shape}, y shape: {y.shape}")
        print(f"x mean: {x.mean():.4f}, x std: {x.std():.4f}")
        print(f"y mean: {y.mean():.4f}, y std: {y.std():.4f}")
        break
    
    print("\n" + "=" * 50)
    print("Testing Rosenbrock (2D standard)")
    print("=" * 50)
    
    train_loader, test_loader = get_regression_data_loader(
        dataset='rosenbrock_2d',
        input_dim=2,
        output_dim=1,
        num_samples=1000,
        normalize=True,
        standardize=True
    )
    
    for x, y in train_loader:
        print(f"x shape: {x.shape}, y shape: {y.shape}")
        print(f"x mean: {x.mean():.4f}, x std: {x.std():.4f}")
        print(f"y mean: {y.mean():.4f}, y std: {y.std():.4f}")
        break
    
    print("\n" + "=" * 50)
    print("Your original sequence-style Rosenbrock (for reference)")
    print("=" * 50)
    
    x_seq, y_seq = generate_rosenbrock_sequence(10)
    print(f"x_seq shape: {x_seq.shape}, y_seq shape: {y_seq.shape}")
    print(f"x_seq: {x_seq[:5]}")
    print(f"y_seq: {y_seq[:5]}")