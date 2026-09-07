import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

# Default configurations
regression = np.array([(1, 3)])
classification = []

parser = argparse.ArgumentParser(description='PyTorch CIFAR training')
parser.add_argument('--hidden_dim', type=int, default=20, metavar='N',
                    help='hidden dimension for the simple model (default: 20)')

class SpLinearBlock(nn.Module):
    """Linear block with splitting-aware forward pass - matches author's style"""
    
    def __init__(self, in_features, out_features, activation='relu', dummy=False):
        super(SpLinearBlock, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.activation = activation
        self.apply_dummy = dummy
        self.dummy = []
        
    def _swish(self, x):
        return x * torch.sigmoid(x)
    
    def _d_swish(self, x):
        """First derivative of swish"""
        s = torch.sigmoid(x)
        return s + x * s * (1. - s)
    
    def _dd_swish(self, x):
        """Second derivative of swish"""
        s = torch.sigmoid(x)
        return s * (1. - s) + s + x * s * (1. - s) - (s**2 + 2. * x * s**2 * (1. - s))
    
    def _dd_softplus(self, x, beta=3.):
        """Second derivative of softplus (smooth ReLU approximation)"""
        z = x * beta
        o = torch.sigmoid(z)
        return beta * o * (1. - o)
    
    def _activate(self, x):
        """Apply activation function"""
        if self.activation is None:
            return x
        if self.activation == 'swish':
            return self._swish(x)
        return F.relu(x)
    
    def forward(self, x):
        """Regular forward pass"""
        out = self.linear(x)
        return self._activate(out)
    
    def sp_forward(self, x):
        """
        Splitting-aware forward pass.
        This matches the author's implementation in sp_conv.py for FC layers.
        
        The key insight from the authors:
        - Use zeros for dummy variables (they get gradients via backprop)
        - The dummy term is weighted by the second derivative of activation
        - This computes the Hessian approximation needed for splitting
        """
        linear_out = self.linear(x)  # [batch, out_features]
        
        # Apply activation
        activated = self._activate(linear_out)
        
        if not self.apply_dummy:
            return activated
        
        # Reset dummy list
        self.dummy = []
        batch, dim = x.shape  # dim = in_features
        n_out = linear_out.shape[1]  # out_features
        
        # Compute second derivative of activation function
        if self.activation is None:
            act_sec_ord_grad = torch.ones_like(linear_out)  # [batch, out_features]
        elif self.activation == 'swish':
            act_sec_ord_grad = self._dd_swish(linear_out)  # [batch, out_features]
        else:
            act_sec_ord_grad = self._dd_softplus(linear_out)  # [batch, out_features]
        
        # Reshape for computation
        act_sec_ord_grad = act_sec_ord_grad.permute(1, 0)  # [out_features, batch]
        
        # Create dummy variables for each output neuron
        # Following the authors: use zeros, NOT random!
        aux_terms = []
        
        for i in range(n_out):
            # Dummy variable for the i-th output neuron
            # Shape: [in_features, in_features] (matches author's dim x dim)
            V = Variable(torch.zeros(dim, dim, device=x.device), requires_grad=True)
            self.dummy.append(V)
            
            # Compute x^T V x for each sample
            # This is the FC version of the authors' patch-based computation
            tmp = torch.matmul(x, V)  # [batch, dim]
            dummy_term = (tmp * x).sum(dim=1, keepdim=True)  # [batch, 1]
            
            # Weight by the second derivative of activation
            # This is the key difference from a simple linearization
            left = act_sec_ord_grad[i:i+1, :]  # [1, batch]
            weighted_dummy_term = left * dummy_term.permute(1, 0)  # [1, batch]
            aux_terms.append(weighted_dummy_term)
        
        # Combine all auxiliary terms
        aux = torch.cat(aux_terms, dim=0).permute(1, 0)  # [batch, out_features]
        
        # Return activated + auxiliary term
        # This matches the authors' pattern: out = activation(bn_out) + aux
        return activated + aux

class SimpleModel(nn.Module):
    """Simple model with splitting capabilities - matches author's style"""
    
    def __init__(self, cfg=None, dataset='cifar10', activation='relu', dummy_layer=-1):
        super(SimpleModel, self).__init__()
        if cfg is None:
            if dataset == 'rosenbrock' or dataset == 'rastrin':
                self.cfg = regression
            else:
                self.cfg = classification
        else:
            self.cfg = cfg
            print("Using custom cfg:", self.cfg)
            
        self.dummy_layer = dummy_layer
        self.activation = activation
        self.layers = self._make_layers()
        
        if dataset == 'rosenbrock' or dataset == 'rastrin':
            self.linear = nn.Linear(self.cfg[-1][1], 1)
        else:
            raise NotImplementedError
        
    def _make_layers(self):
        """Create layers with dummy flag for the specified layer"""
        layers = []
        for idx, (inp, out) in enumerate(self.cfg):
            # Only apply dummy to the specified layer
            dummy = (idx == self.dummy_layer)
            # No activation on final layer (matches author's pattern)
            activation = self.activation if idx < len(self.cfg) - 1 else None
            layers.append(SpLinearBlock(inp, out, activation=activation, dummy=dummy))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        """Standard forward pass"""
        out = x
        for layer in self.layers:
            out = layer(out)
        out = self.linear(out)  # Final layer without activation
        return out
    
    def sp_forward(self, x):
        """
        Splitting-aware forward pass.
        This is called by main_finetune.py's compute_A()
        """
        out = x
        for layer in self.layers:
            out = layer.sp_forward(out)
        out = self.linear(out)  # Final layer without activation
        return out

def save_model(model, opt, epoch, filename, path):
    """Save model checkpoint"""
    pth = {
        'state_dict': model.state_dict(),
        'intermediate_layers': model.intermediate_layers if hasattr(model, 'intermediate_layers') else [],
        'model_structure': get_model_details(model),
        'optimizer_name': opt.__class__.__name__,
        'optimizer': opt.state_dict(),
        'epoch': epoch
    }
    torch.save(pth, f"{path}/{filename}.pth")
    
def get_model_details(model):
    """Get model architecture details"""
    return {
        'cfg': model.cfg if hasattr(model, 'cfg') else [],
        'dummy_layer': model.dummy_layer if hasattr(model, 'dummy_layer') else -1,
        'activation': model.activation if hasattr(model, 'activation') else 'relu'
    }

if __name__ == "__main__":
    print("Output dimension:", regression[-1][1])
    
    # Test the model
    cfg = [(1, 3)]
    model = SimpleModel(cfg=cfg, dataset='rosenbrock', activation='relu', dummy_layer=0)
    print("Model created successfully!")
    print("Model details:", get_model_details(model))
    
    # Test forward pass
    x = torch.randn(4, 1)
    print("Input shape:", x.shape)
    
    # Regular forward
    y = model(x)
    print("Regular forward output shape:", y.shape)
    
    # Splitting-aware forward
    y_sp = model.sp_forward(x)
    print("Splitting forward output shape:", y_sp.shape)
    
    # Check dummy variables
    for i, layer in enumerate(model.layers):
        if hasattr(layer, 'dummy') and len(layer.dummy) > 0:
            print(f"Layer {i} has {len(layer.dummy)} dummy variables")
            print(f"  dummy[0].shape: {layer.dummy[0].shape}")
            print(f"  dummy[0].requires_grad: {layer.dummy[0].requires_grad}")
    
    # Test gradients (simulating compute_A)
    print("\n--- Simulating compute_A ---")
    loss = y_sp.sum()
    loss.backward()
    
    print("Gradients after backward:")
    for i, layer in enumerate(model.layers):
        if hasattr(layer, 'dummy') and len(layer.dummy) > 0:
            for j, param in enumerate(layer.dummy):
                if param.grad is not None:
                    grad_norm = param.grad.norm().item()
                    print(f"  Layer {i}, dummy {j}: grad norm = {grad_norm:.6f}")
                    if grad_norm > 1e-8:
                        print(f"    ✅ Gradient is non-zero!")
                    else:
                        print(f"    ⚠️ Gradient is near zero!")
                else:
                    print(f"  Layer {i}, dummy {j}: grad is None ❌")
    
    # Check if A matrix would be valid
    print("\n--- Checking A matrix ---")
    A = []
    for param in model.layers[0].dummy:
        if param.grad is not None:
            A.append(param.grad.data.cpu().numpy())
    
    if len(A) > 0:
        A = np.array(A)
        print(f"A shape: {A.shape}")
        print(f"A mean: {A.mean():.6f}")
        print(f"A std: {A.std():.6f}")
        
        if not np.allclose(A, 0):
            print("✅ A has non-zero values!")
        else:
            print("❌ A is all zeros!")
    else:
        print("❌ No gradients collected!")