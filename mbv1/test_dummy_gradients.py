import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

from model_custom import SimpleModel

def test_dummy_gradients():
    print("=" * 60)
    print("TESTING DUMMY VARIABLE GRADIENTS")
    print("=" * 60)
    
    # Create model with cfg
    cfg = [(1, 3)]  # One hidden layer: 1 -> 3
    model = SimpleModel(cfg=cfg, dataset='rosenbrock', activation='relu', dummy_layer=0)
    
    # Create input
    x = torch.randn(4, 1, requires_grad=True)
    print(f"Input shape: {x.shape}")
    print(f"Input requires_grad: {x.requires_grad}")
    
    # Forward pass
    print("\n--- Forward pass ---")
    output = model.sp_forward(x)
    print(f"Output shape: {output.shape}")
    print(f"Output requires_grad: {output.requires_grad}")
    
    # Check dummy variables in the layer
    layer = model.layers[0]
    print(f"\n--- Dummy variables ---")
    print(f"Number of dummy variables: {len(layer.dummy)}")
    for i, dummy in enumerate(layer.dummy):
        print(f"  dummy[{i}].shape: {dummy.shape}")
        print(f"  dummy[{i}].requires_grad: {dummy.requires_grad}")
        if dummy.grad is not None:
            print(f"  dummy[{i}].grad.shape: {dummy.grad.shape}")
        else:
            print(f"  dummy[{i}].grad: None (before backward)")
    
    # Compute loss
    print("\n--- Loss computation ---")
    # For regression, MSE loss
    target = torch.randn(4, 1)
    loss = F.mse_loss(output, target)
    print(f"Loss value: {loss.item():.6f}")
    print(f"Loss requires_grad: {loss.requires_grad}")
    
    # Backward pass
    print("\n--- Backward pass ---")
    loss.backward()
    print("Backward complete!")
    
    # Check gradients after backward
    print("\n--- Gradients after backward ---")
    for i, dummy in enumerate(layer.dummy):
        if dummy.grad is not None:
            grad_norm = dummy.grad.norm().item()
            grad_mean = dummy.grad.mean().item()
            print(f"  dummy[{i}].grad:")
            print(f"    shape: {dummy.grad.shape}")
            print(f"    norm: {grad_norm:.6f}")
            print(f"    mean: {grad_mean:.6f}")
            if grad_norm > 1e-8:
                print(f"    ✅ Gradient is non-zero! dummy variable is working.")
            else:
                print(f"    ⚠️ Gradient is near zero! dummy may not affect loss.")
        else:
            print(f"  dummy[{i}].grad: None ❌")
            print(f"    ❌ No gradient! dummy is not connected to loss.")
    
    # Also check the input gradient
    if x.grad is not None:
        print(f"\nInput gradient norm: {x.grad.norm().item():.6f}")
    
    # Check if aux term actually affects the loss
    print("\n--- Verifying dummy contribution ---")
    with torch.no_grad():
        # Forward without dummy
        model.layers[0].apply_dummy = False
        output_no_dummy = model.sp_forward(x)
        model.layers[0].apply_dummy = True
        
        diff = (output - output_no_dummy).abs().mean().item()
        print(f"Mean difference between outputs with/without dummy: {diff:.6f}")
        if diff > 1e-8:
            print(f"  ✅ Dummy variables affect the output!")
        else:
            print(f"  ⚠️ Dummy variables have no effect on output!")

if __name__ == "__main__":
    test_dummy_gradients()