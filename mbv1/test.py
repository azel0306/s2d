import matplotlib.pyplot as plt
import numpy as np
import torch

# 1D Rastrigin
x = torch.linspace(-5, 5, 1000)
A = 10
y = A + x**2 - A * torch.cos(2 * np.pi * x)

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(x, y)
plt.title('1D Rastrigin Function')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.grid(True)

# 2D Rastrigin
X1, X2 = torch.meshgrid(torch.linspace(-5, 5, 100), torch.linspace(-5, 5, 100))
Z = 2*A + X1**2 + X2**2 - A*torch.cos(2*np.pi*X1) - A*torch.cos(2*np.pi*X2)

plt.subplot(1, 2, 2)
plt.contourf(X1, X2, Z, levels=50, cmap='viridis')
plt.title('2D Rastrigin Function')
plt.xlabel('x1')
plt.ylabel('x2')
plt.colorbar()
plt.tight_layout()
plt.show()