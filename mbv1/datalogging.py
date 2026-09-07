import pandas as pd
import torch


dct = {'train_acc': [], 'train_loss': [], 'test_acc': [], 'test_loss': []}

for idx in range(20):
    dct['train_acc'].append(torch.rand(1).item())
    dct['train_loss'].append(torch.rand(1).item())
    dct['test_acc'].append(torch.rand(1).item())
    dct['test_loss'].append(torch.rand(1).item())
    
df = pd.DataFrame(dct)
print(df.head())

df.to_csv('test.csv', index=False)