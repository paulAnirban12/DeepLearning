import torch
import torch.nn as nn

def BN_forward(x,gamma,beta,error=1e-5):
  miu = x.mean()
  variance = ((x-miu)**2).mean()
  x_hat = (x-miu)/torch.sqrt(variance+error)
  y = gamma*x_hat + beta
  return y



X = torch.tensor([2., 4., 6., 8.])

bn = nn.BatchNorm1d(1, affine=True)
bn.weight.data.fill_(1.5)  # gamma
bn.bias.data.fill_(0.5)    # beta
bn.eval()

y_torch = bn(X.view(-1,1)).view(-1)

gamma = torch.tensor(1.5)
beta = torch.tensor(0.5)

y_manual = BN_forward(X, gamma, beta)

print("Manual BN Output:", y_manual)
print("PyTorch BN Output:",y_torch)

