import torch
import torch.nn as nn
import torch.nn.functional as F

class LaplacianGPAF(nn.Module):
    def __init__(self, k=2):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(0.1))
        self.beta = nn.Parameter(torch.tensor(1.0))
        self.gamma = nn.Parameter(torch.tensor(0.1))
        self.k = k

        # Define Laplacian kernel
        laplacian_kernel = torch.tensor([[0, 1, 0],
                                         [1, -4, 1],
                                         [0, 1, 0]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        self.register_buffer('laplacian_kernel', laplacian_kernel)

    def forward(self, y):
        # Expand kernel to match input channels
        laplacian_kernel = self.laplacian_kernel.expand(y.shape[1], 1, 3, 3)
        lap = F.conv2d(y, laplacian_kernel, padding=1, groups=y.shape[1])
        out = y + self.alpha * lap + self.beta * y + self.gamma * (y ** self.k)
        return out

class KGActivation(nn.Module):
    def __init__(self):
        super().__init__()
        # Initialize parameters as learnable scalars
        self.alpha = nn.Parameter(torch.tensor(1.0))
        self.beta = nn.Parameter(torch.tensor(1.0))
        self.gamma = nn.Parameter(torch.tensor(0.0))

    def forward(self, x):
        return self.alpha * torch.cos(self.beta * x) + self.gamma * torch.sin(self.beta * x)