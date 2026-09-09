import torch
import torch.nn as nn
import torch.optim as optim
from math import pi
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

torch.manual_seed(20260908)  # For reproducibility
# CONSTANTS
dataset_name = 'rastrigin'  # Change to 'xsquare' or 'cosine' for other datasets
batch_size = 256
epochs = 400

if dataset_name == 'rastrigin':
    x_min = 0
    x_max = pi/2
    y_padding = 5
    noise_std = 0.0
    num_samples = 256
elif dataset_name == 'xsquare':
    x_min = -5.0
    x_max = 5.0
    y_padding = 2
    noise_std = 0.0
    num_samples = 256
elif dataset_name == 'cosine':
    x_min = 0
    x_max = 2*pi
    y_padding = 0.5
    noise_std = 0.0
    num_samples = 256

# ==================== DATA GENERATION ====================

def generate_rastrigin(num_samples=256, x_min=-5.12, x_max=5.12, noise_std=0.02):
    """Generate 1D Rastrigin dataset with noise only in inputs."""
    x_true = torch.rand(num_samples) * (x_max - x_min) + x_min
    x_noisy = x_true + torch.randn_like(x_true) * noise_std * (x_max - x_min)
    y = 10 + x_noisy**2 - 10 * torch.cos(2 * np.pi * x_noisy)
    return x_noisy.unsqueeze(1).float(), y.unsqueeze(1).float()

def generate_xsquare(num_samples=256, x_min=-2.0, x_max=2.0, noise_std=0.02):
    """Generate 1D X^2 dataset with noise."""
    x_true = torch.rand(num_samples) * (x_max - x_min) + x_min
    x_noisy = x_true + torch.randn_like(x_true) * noise_std * (x_max - x_min)
    y = x_noisy**2
    return x_noisy.unsqueeze(1).float(), y.unsqueeze(1).float()

def generate_cosine(num_samples=256, x_min=-2.0, x_max=2.0, noise_std=0.02):
    """Generate 1D Cosine dataset with noise."""
    x_true = torch.rand(num_samples) * (x_max - x_min) + x_min
    x_noisy = x_true + torch.randn_like(x_true) * noise_std * (x_max - x_min)
    y = torch.cos(x_noisy)
    return x_noisy.unsqueeze(1).float(), y.unsqueeze(1).float()

# ==================== ARCHITECTURES ====================

def create_architecture(name, hidden_size=64, num_layers=2):
    """Create different model architectures."""
    if name == 'Shallow_8':
        return nn.Sequential(
            nn.Linear(1, 8),
            nn.ReLU(),
            nn.Linear(8, 1)
        )
    elif name == 'Shallow_32':
        return nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    elif name == 'Shallow_64':
        return nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    
    elif name == 'Shallow_256':
        return nn.Sequential(
            nn.Linear(1, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
    
    elif name == 'Shallow_512':
        return nn.Sequential(
            nn.Linear(1, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )
    
    elif name == 'Deep_64x2':
        return nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    
    elif name == 'Deep_64x3':
        return nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    
    elif name == 'Deep_64x4':
        return nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    
    elif name == 'Deep_128x3':
        return nn.Sequential(
            nn.Linear(1, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
    
    elif name == 'LeakyReLU_64x3':
        return nn.Sequential(
            nn.Linear(1, 64),
            nn.LeakyReLU(0.01),
            nn.Linear(64, 64),
            nn.LeakyReLU(0.01),
            nn.Linear(64, 64),
            nn.LeakyReLU(0.01),
            nn.Linear(64, 1)
        )
    
    elif name == 'Tanh_64x3':
        return nn.Sequential(
            nn.Linear(1, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )
    
    elif name == 'WithBatchNorm_64x3':
        return nn.Sequential(
            nn.Linear(1, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    
    else:
        raise ValueError(f"Unknown architecture: {name}")

# ==================== TRAINING FUNCTION ====================

def train_model(model, train_loader, test_loader, epochs=200, lr=0.0005, device='cpu'):
    """Train a single model and return losses."""
    
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=20, factor=0.5)
    
    train_losses = []
    test_losses = []
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_batches = 0
        
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            y_pred = model(x_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            
            # Gradient clipping (important without normalization)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            train_loss += loss.item()
            train_batches += 1
        
        avg_train_loss = train_loss / train_batches
        train_losses.append(avg_train_loss)
        
        # Testing
        model.eval()
        test_loss = 0.0
        test_batches = 0
        
        with torch.no_grad():
            for x_batch, y_batch in test_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                y_pred = model(x_batch)
                loss = criterion(y_pred, y_batch)
                test_loss += loss.item()
                test_batches += 1
        
        avg_test_loss = test_loss / test_batches
        test_losses.append(avg_test_loss)
        
        # Learning rate scheduling
        scheduler.step(avg_test_loss)
    
    return model, train_losses, test_losses

# ==================== PREDICTION VISUALIZATION ====================

def plot_predictions(models_dict, x_test_plot, y_true, title_prefix="", device='cpu'):
    """Plot predictions vs ground truth for multiple models."""
    
    n_models = len(models_dict)
    n_cols = min(3, n_models)
    n_rows = (n_models + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
    if n_models == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    # Move input to the correct device
    x_test_plot = x_test_plot.to(device)
    y_true = y_true.to(device)
    
    for idx, (name, model) in enumerate(models_dict.items()):
        ax = axes[idx]
        
        # Get predictions
        model.eval()
        with torch.no_grad():
            y_pred = model(x_test_plot)
        
        # Move back to CPU for plotting
        x_test_plot_cpu = x_test_plot.cpu().numpy()
        y_true_cpu = y_true.cpu().numpy()
        y_pred_cpu = y_pred.cpu().numpy()
        
        # Calculate metrics
        mse = nn.MSELoss()(y_pred, y_true).item()
        rmse = np.sqrt(mse)
        mae = nn.L1Loss()(y_pred, y_true).item()
        
        # Plot
        ax.plot(x_test_plot_cpu, y_true_cpu, 'b-', label='Ground Truth', linewidth=2, alpha=0.8)
        ax.plot(x_test_plot_cpu, y_pred_cpu, 'r--', label=f'{name}', linewidth=2, alpha=0.8)
        
        # Add error band (±RMSE)
        ax.fill_between(x_test_plot_cpu.flatten(), 
                       (y_true_cpu - rmse).flatten(),
                       (y_true_cpu + rmse).flatten(),
                       alpha=0.2, color='red', label=f'±RMSE ({rmse:.3f})')
        
        ax.set_xlabel('x')
        ax.set_ylabel('f(x)')
        ax.set_title(f'{name}\nMSE: {mse:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(x_min-0.5, x_max+0.5)
        ax.set_ylim(y_true.min().item() - y_padding, y_true.max().item() + y_padding)
    
    # Hide empty subplots
    for idx in range(len(models_dict), len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle(f'{title_prefix} Predictions vs Ground Truth', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.show()

def plot_comprehensive_comparison(results, x_test_plot, y_true, device='cpu'):
    """Create comprehensive prediction comparison visualization."""
    
    # Sort by performance
    sorted_results = sorted(results.items(), key=lambda x: x[1]['final_test_loss'])
    
    # Move to device
    x_test_plot = x_test_plot.to(device)
    y_true = y_true.to(device)
    
    # 1. Best vs Worst
    best_name, best_result = sorted_results[0]
    worst_name, worst_result = sorted_results[-1]
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # 1a. Best model prediction
    ax = axes[0, 0]
    best_model = best_result['model']
    best_model.eval()
    with torch.no_grad():
        y_pred_best = best_model(x_test_plot)
    
    # Move to CPU for plotting
    x_test_cpu = x_test_plot.cpu().numpy()
    y_true_cpu = y_true.cpu().numpy()
    y_pred_best_cpu = y_pred_best.cpu().numpy()
    
    mse_best = nn.MSELoss()(y_pred_best, y_true).item()
    rmse_best = np.sqrt(mse_best)
    
    ax.plot(x_test_cpu, y_true_cpu, 'b-', label='Ground Truth', linewidth=2)
    ax.plot(x_test_cpu, y_pred_best_cpu, 'r--', label='Best Model Prediction', linewidth=2)
    ax.fill_between(x_test_cpu.flatten(),
                   (y_true_cpu - rmse_best).flatten(),
                   (y_true_cpu + rmse_best).flatten(),
                   alpha=0.2, color='green')
    ax.set_xlabel('x')
    ax.set_ylabel('f(x)')
    ax.set_title(f'Best Model: {best_name}\nMSE: {mse_best:.4f}, RMSE: {rmse_best:.4f}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 1b. Worst model prediction
    ax = axes[0, 1]
    worst_model = worst_result['model']
    worst_model.eval()
    with torch.no_grad():
        y_pred_worst = worst_model(x_test_plot)
    
    y_pred_worst_cpu = y_pred_worst.cpu().numpy()
    mse_worst = nn.MSELoss()(y_pred_worst, y_true).item()
    rmse_worst = np.sqrt(mse_worst)
    
    ax.plot(x_test_cpu, y_true_cpu, 'b-', label='Ground Truth', linewidth=2)
    ax.plot(x_test_cpu, y_pred_worst_cpu, 'r--', label='Worst Model Prediction', linewidth=2)
    ax.fill_between(x_test_cpu.flatten(),
                   (y_true_cpu - rmse_worst).flatten(),
                   (y_true_cpu + rmse_worst).flatten(),
                   alpha=0.2, color='red')
    ax.set_xlabel('x')
    ax.set_ylabel('f(x)')
    ax.set_title(f'Worst Model: {worst_name}\nMSE: {mse_worst:.4f}, RMSE: {rmse_worst:.4f}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 1c. Error comparison
    ax = axes[0, 2]
    errors_best = (y_pred_best - y_true).cpu().numpy().flatten()
    errors_worst = (y_pred_worst - y_true).cpu().numpy().flatten()
    
    ax.hist(errors_best, bins=30, alpha=0.5, label=f'Best ({best_name})', color='green', density=True)
    ax.hist(errors_worst, bins=30, alpha=0.5, label=f'Worst ({worst_name})', color='red', density=True)
    ax.axvline(x=0, color='black', linestyle='--', linewidth=0.5)
    ax.set_xlabel('Prediction Error')
    ax.set_ylabel('Density')
    ax.set_title('Error Distribution Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. All models predictions overlay
    ax = axes[1, 0]
    ax.plot(x_test_cpu, y_true_cpu, 'k-', label='Ground Truth', linewidth=2)
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(sorted_results)))
    for idx, (name, result) in enumerate(sorted_results):
        model = result['model']
        model.eval()
        with torch.no_grad():
            y_pred = model(x_test_plot)
        y_pred_cpu = y_pred.cpu().numpy()
        ax.plot(x_test_cpu, y_pred_cpu, '--', color=colors[idx], 
                label=f'{name} ({result["final_test_loss"]:.3f})', alpha=0.6, linewidth=1)
    
    ax.set_xlabel('x')
    ax.set_ylabel('f(x)')
    ax.set_title('All Models Predictions Overlay')
    ax.legend(loc='upper right', fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)
    
    # 2b. Error vs x for all models
    ax = axes[1, 1]
    for idx, (name, result) in enumerate(sorted_results[:5]):  # Show top 5
        model = result['model']
        model.eval()
        with torch.no_grad():
            y_pred = model(x_test_plot)
        errors = (y_pred - y_true).cpu().numpy().flatten()
        ax.scatter(x_test_cpu.flatten(), errors, 
                  s=5, alpha=0.5, label=f'{name} ({result["final_test_loss"]:.3f})')
    
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.set_xlabel('x')
    ax.set_ylabel('Error')
    ax.set_title('Error vs Input (Top 5 Models)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # 2c. Prediction scatter for best model
    ax = axes[1, 2]
    y_true_scatter = y_true_cpu.flatten()
    y_pred_scatter = y_pred_best_cpu.flatten()
    
    ax.scatter(y_true_scatter, y_pred_scatter, s=10, alpha=0.5, c='blue')
    ax.plot([0, 45], [0, 45], 'r--', linewidth=2, label='Perfect Prediction')
    ax.set_xlabel('True Values')
    ax.set_ylabel('Predicted Values')
    ax.set_title(f'Best Model: {best_name}\nPrediction vs Ground Truth')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-2, 45)
    ax.set_ylim(-2, 45)
    
    plt.suptitle('Comprehensive Prediction Comparison', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.show()

# ==================== MAIN COMPARISON ====================

def compare_architectures():
    """Compare all architectures on Rastrigin without normalization."""
    
    print("=" * 70)
    print(f"COMPARING ARCHITECTURES ON {dataset_name.upper()} (WITHOUT NORMALIZATION)")
    print("=" * 70)
    
    # Generate data
    
    print(f"\n📊 Generating {dataset_name.upper()} data...")
    if dataset_name == 'rastrigin':
        x, y = generate_rastrigin(num_samples=num_samples, x_min=x_min, x_max=x_max, noise_std=noise_std)
    elif dataset_name == 'xsquare':
        x, y = generate_xsquare(num_samples=num_samples, x_min=x_min, x_max=x_max, noise_std=noise_std)
    elif dataset_name == 'cosine':
        x, y = generate_cosine(num_samples=num_samples, x_min=x_min, x_max=x_max, noise_std=noise_std)
    print(f"x range: [{x.min():.3f}, {x.max():.3f}]")
    print(f"y range: [{y.min():.3f}, {y.max():.3f}]")
    
    # Split data
    dataset = TensorDataset(x, y)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Architectures to compare
    architectures = [
        'Shallow_8',
        'Shallow_32',
        'Shallow_64',
        'Shallow_256',
        'Shallow_512',
        'Deep_64x2',
        'Deep_64x3',
        'Deep_64x4',
        'Deep_128x3',
        'LeakyReLU_64x3',
        'Tanh_64x3',
        'WithBatchNorm_64x3'
    ]
    
    # Train each architecture
    results = {}
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n🚀 Using device: {device}")
    
    for arch_name in tqdm(architectures, desc="Training architectures"):
        # Create model
        model = create_architecture(arch_name)
        total_params = sum(p.numel() for p in model.parameters())
        
        # Adjust learning rate based on architecture
        if 'Shallow' in arch_name:
            lr = 0.0005
        elif 'WithBatchNorm' in arch_name:
            lr = 0.001
        else:
            lr = 0.0005
        
        # Train
        model, train_losses, test_losses = train_model(
            model, train_loader, test_loader, 
            epochs=epochs, lr=lr, device=device
        )
        
        # Store results
        results[arch_name] = {
            'model': model,
            'train_losses': train_losses,
            'test_losses': test_losses,
            'final_train_loss': train_losses[-1],
            'final_test_loss': test_losses[-1],
            'params': total_params
        }
        
        print(f"  {arch_name:15s} | Params: {total_params:6d} | Test MSE: {test_losses[-1]:.4f}")
    
    # ==================== VISUALIZATION ====================
    
    # Generate test data for visualization
    x_test_plot = torch.linspace(x_min, x_max, num_samples).unsqueeze(1)
    if dataset_name == 'rastrigin':
        y_true = 10 + x_test_plot**2 - 10 * torch.cos(2 * np.pi * x_test_plot)
    elif dataset_name == 'xsquare':
        y_true = x_test_plot**2
    elif dataset_name == 'cosine':
        y_true = torch.cos(x_test_plot)

    # 1. Training curves comparison
    fig, axes = plt.subplots(3, 4, figsize=(20, 10))
    axes = axes.flatten()
    
    for idx, (name, result) in enumerate(results.items()):
        ax = axes[idx]
        ax.plot(result['train_losses'], label='Train', color='blue', alpha=0.7)
        ax.plot(result['test_losses'], label='Test', color='red', alpha=0.7)
        ax.set_title(f'{name}\nTest MSE: {result["final_test_loss"]:.4f}\nParams: {result["params"]}')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('MSE Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')
    
    plt.tight_layout()
    plt.show()
    
    # 2. Bar chart of final test losses
    fig, ax = plt.subplots(figsize=(12, 6))
    
    names = list(results.keys())
    test_losses = [results[name]['final_test_loss'] for name in names]
    colors = ['red' if loss > 20 else 'orange' if loss > 10 else 'green' for loss in test_losses]
    
    bars = ax.bar(names, test_losses, color=colors, edgecolor='black', alpha=0.7)
    ax.set_xlabel('Architecture')
    ax.set_ylabel('Final Test MSE')
    ax.set_title('Architecture Comparison on Rastrigin (Without Normalization)')
    ax.set_yscale('log')
    
    # Add value labels on bars
    for bar, loss in zip(bars, test_losses):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{loss:.3f}',
                ha='center', va='bottom', fontsize=9)
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()
    
    # 3. Individual predictions for all models
    print("\n📊 Plotting individual predictions...")
    models_dict = {name: result['model'] for name, result in results.items()}
    plot_predictions(models_dict, x_test_plot, y_true, 
                    "Individual Model", device=device)
    
    # # 4. Comprehensive comparison
    # print("\n📊 Plotting comprehensive comparison...")
    # plot_comprehensive_comparison(results, x_test_plot, y_true, device=device)
    
    # # 5. Best model detailed analysis
    # best_name = min(results.keys(), key=lambda k: results[k]['final_test_loss'])
    # best_model = results[best_name]['model']
    # best_loss = results[best_name]['final_test_loss']
    
    # print(f"\n🏆 Best Model: {best_name}")
    # print(f"   Test MSE: {best_loss:.4f}")
    # print(f"   RMSE: {np.sqrt(best_loss):.4f}")
    # print(f"   Parameters: {results[best_name]['params']:,}")
    
    # # Detailed best model visualization
    # fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # best_model.eval()
    # with torch.no_grad():
    #     y_pred = best_model(x_test_plot.to(device))
    
    # # Move to CPU for plotting
    # x_test_cpu = x_test_plot.cpu().numpy()
    # y_true_cpu = y_true.cpu().numpy()
    # y_pred_cpu = y_pred.cpu().numpy()
    
    # # 5a. Prediction vs truth
    # ax = axes[0, 0]
    # ax.plot(x_test_cpu, y_true_cpu, 'b-', label='Ground Truth', linewidth=2)
    # ax.plot(x_test_cpu, y_pred_cpu, 'r--', label='Prediction', linewidth=2)
    # ax.fill_between(x_test_cpu.flatten(),
    #                (y_true_cpu - np.sqrt(best_loss)).flatten(),
    #                (y_true_cpu + np.sqrt(best_loss)).flatten(),
    #                alpha=0.2, color='green', label=f'±RMSE ({np.sqrt(best_loss):.4f})')
    # ax.set_xlabel('x')
    # ax.set_ylabel('f(x)')
    # ax.set_title(f'Best Model: {best_name}\nMSE: {best_loss:.4f}, RMSE: {np.sqrt(best_loss):.4f}')
    # ax.legend()
    # ax.grid(True, alpha=0.3)
    
    # # 5b. Error distribution
    # ax = axes[0, 1]
    # y_pred = y_pred.cpu()
    # y_true = y_true.cpu()
    # errors = (y_pred - y_true).cpu().numpy().flatten()
    # ax.hist(errors, bins=50, alpha=0.7, edgecolor='black', color='blue', density=True)
    # ax.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero Error')
    # ax.axvline(x=np.mean(errors), color='green', linestyle='--', linewidth=2, 
    #            label=f'Mean: {np.mean(errors):.4f}')
    # ax.set_xlabel('Prediction Error')
    # ax.set_ylabel('Density')
    # ax.set_title(f'Error Distribution\nMean: {np.mean(errors):.4f}, Std: {np.std(errors):.4f}')
    # ax.legend()
    # ax.grid(True, alpha=0.3)
    
    # # 5c. Error vs x
    # ax = axes[0, 2]
    # ax.scatter(x_test_cpu.flatten(), errors, s=5, alpha=0.5, c='red')
    # ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    # ax.axhline(y=np.sqrt(best_loss), color='red', linestyle='--', 
    #            label=f'RMSE: {np.sqrt(best_loss):.4f}')
    # ax.axhline(y=-np.sqrt(best_loss), color='red', linestyle='--')
    # ax.set_xlabel('x')
    # ax.set_ylabel('Error')
    # ax.set_title('Error vs Input x')
    # ax.legend()
    # ax.grid(True, alpha=0.3)
    
    # # 5d. Scatter plot
    # ax = axes[1, 0]
    # ax.scatter(y_true_cpu.flatten(), y_pred_cpu.flatten(), s=5, alpha=0.5, c='blue')
    # ax.plot([0, 45], [0, 45], 'r--', linewidth=2, label='Perfect Prediction')
    # ax.set_xlabel('True Values')
    # ax.set_ylabel('Predicted Values')
    # ax.set_title('Prediction vs Ground Truth Scatter')
    # ax.legend()
    # ax.grid(True, alpha=0.3)
    # ax.set_xlim(-2, 45)
    # ax.set_ylim(-2, 45)
    
    # # 5e. Residual plot
    # ax = axes[1, 1]
    # residuals = (y_pred - y_true).cpu().numpy().flatten()
    # ax.scatter(y_pred_cpu.flatten(), residuals, s=5, alpha=0.5, c='purple')
    # ax.axhline(y=0, color='red', linestyle='--', linewidth=2)
    # ax.set_xlabel('Predicted Values')
    # ax.set_ylabel('Residuals')
    # ax.set_title('Residual Plot')
    # ax.grid(True, alpha=0.3)
    
    # # 5f. Q-Q plot
    # ax = axes[1, 2]
    # from scipy import stats
    # stats.probplot(residuals, dist="norm", plot=ax)
    # ax.set_title('Q-Q Plot of Residuals')
    # ax.grid(True, alpha=0.3)
    
    # plt.suptitle(f'Best Model Detailed Analysis: {best_name}', fontsize=14, y=1.02)
    # plt.tight_layout()
    # plt.show()
    
    # # 6. Summary table
    # fig, ax = plt.subplots(figsize=(14, 8))
    # ax.axis('tight')
    # ax.axis('off')
    
    # sorted_results = sorted(results.items(), key=lambda x: x[1]['final_test_loss'])
    
    # table_data = []
    # for idx, (name, result) in enumerate(sorted_results, 1):
    #     # Get predictions for this model
    #     model = result['model']
    #     model.eval()
    #     with torch.no_grad():
    #         y_pred = model(x_test_plot.to(device))
    #     mse = nn.MSELoss()(y_pred, y_true.to(device)).item()
    #     rmse = np.sqrt(mse)
    #     mae = nn.L1Loss()(y_pred, y_true.to(device)).item()
    #     r2 = 1 - (mse / torch.var(y_true.to(device)).item())
        
    #     table_data.append([
    #         idx,
    #         name,
    #         f"{result['params']:,}",
    #         f"{result['final_train_loss']:.4f}",
    #         f"{result['final_test_loss']:.4f}",
    #         f"{rmse:.4f}",
    #         f"{mae:.4f}",
    #         f"{r2:.4f}"
    #     ])
    
    # columns = ['Rank', 'Architecture', 'Params', 'Train MSE', 'Test MSE', 'RMSE', 'MAE', 'R²']
    # table = ax.table(cellText=table_data, colLabels=columns, 
    #                  cellLoc='center', loc='center',
    #                  colColours=['#f0f0f0']*8)
    # table.auto_set_font_size(False)
    # table.set_fontsize(10)
    # ax.set_title('All Models Comparison Table', fontsize=14, pad=20)
    
    # plt.tight_layout()
    # plt.show()
    
    return results, best_name

# ==================== RUN COMPARISON ====================

if __name__ == "__main__":
    results, best_name = compare_architectures()
    
    print("\n" + "=" * 70)
    print("✅ COMPARISON COMPLETE")
    print("=" * 70)
    print(f"\n🏆 Best Architecture: {best_name}")
    print(f"   Test MSE: {results[best_name]['final_test_loss']:.4f}")
    print(f"   Parameters: {results[best_name]['params']:,}")
    print(f"   RMSE: {np.sqrt(results[best_name]['final_test_loss']):.4f}")