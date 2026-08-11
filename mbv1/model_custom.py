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
        
        # Initialize dummy as empty list - matches main_finetune expectation
        self.dummy = []
        
    def _swish(self, x):
        return x * torch.sigmoid(x)
    
    def _dd_swish(self, x):
        """Second derivative of swish"""
        s = torch.sigmoid(x)
        return s * (1 - s) * (2 * x + 1)
    
    def _dd_softplus(self, x, beta=3.):
        """Second derivative of softplus (smooth ReLU approximation)"""
        z = x * beta
        o = torch.sigmoid(z)
        return beta * o * (1 - o)
    
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
        This matches the author's implementation in sp_conv.py
        """
        linear_out = self.linear(x)
        activated = self._activate(linear_out)
        
        if not self.apply_dummy:
            return activated
        
        # Reset dummy list for this forward pass
        self.dummy = []
        batch, dim = x.shape
        aux_terms = []
        
        for i in range(linear_out.shape[1]):
            V = Variable(torch.zeros(dim, dim, device=x.device), requires_grad=True)
            self.dummy.append(V)
            
            tmp = torch.matmul(x, V)
            dummy_term = (tmp * x).sum(dim=1, keepdim=True)
            aux_terms.append(dummy_term)
        
        aux = torch.cat(aux_terms, dim=1)
        return activated + aux

class SimpleModel(nn.Module):
    """Simple model with splitting capabilities - matches author's style"""
    
    def __init__(self, cfg=None, dataset='cifar10', activation='relu', dummy_layer=-1):
        super(SimpleModel, self).__init__()
        if cfg is None:
            if dataset == 'rosenbrock' or dataset == 'rastrin':
                print("#"*64)
                self.cfg = regression
            else:
                self.cfg = classification ## TODO 
        else:
            self.cfg = cfg
            print("Using custom cfg:", self.cfg)
            
        self.dummy_layer = dummy_layer
        self.activation = activation
        self.layers = self._make_layers()
        
        if dataset == 'rosenbrock' or dataset == 'rastrin':
            self.linear = nn.Linear(self.cfg[-1][1], 1)  
        else: 
            NotImplementedError
        
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
    cfg = [(1, 3), (3, 1)]  # Example configuration
    print(cfg[:-1])
    model = SimpleModel(cfg=cfg, dataset='custom', activation='relu', dummy_layer=0)
    print("Model created successfully!")
    print("Model details:", get_model_details(model))
    print("Model structure:", model)
    
    # Test forward pass
    x = torch.randn(10, 1)
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
            print(f"dummy[0] requires_grad: {layer.dummy[0].requires_grad}")
            
    for name, mm in model.named_children():
        if name == 'layers':
            print(f"Layers: {mm}")
            for i, layer in enumerate(mm):
                print(f"Layer {i}: {layer}")
        elif name == 'linear':
            print(f"Final linear layer: {mm}")