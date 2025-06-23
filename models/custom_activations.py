import torch
import torch.nn as nn
import torch.nn.functional as F


class CustomActivationPlaceholder(nn.Module):
    def __init__(self):
        super().__init__()
        self.act_fn = lambda x: x

    def set_activation(self, act_module):
        self.act_fn = act_module

    def forward(self, x):
        return self.act_fn(x)

class ChannelwiseActivation(nn.Module):
    def __init__(self, activations_per_channel):
        super().__init__()
        assert isinstance(activations_per_channel, list) or isinstance(activations_per_channel, nn.ModuleList), "activations_per_channel must be a list or ModuleList"
        self.activations = nn.ModuleList(activations_per_channel)

    def forward(self, x):
        assert x.shape[1] == len(self.activations), (
            f"Number of activations ({len(self.activations)}) must match number of channels ({x.shape[1]})"
        )
        outs = []
        for c in range(x.shape[1]):
            out = self.activations[c](x[:, c:c+1, :, :])
            outs.append(out)
        return torch.cat(outs, dim=1)


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

channel_map = {
    # Top conv1 activation
    'activation': 64,

    # === Layer 1 ===
    'layer1.0.act1': 64,
    'layer1.0.act2': 64,
    'layer1.0.act3': 256,

    'layer1.1.act1': 64,
    'layer1.1.act2': 64,
    'layer1.1.act3': 256,

    'layer1.2.act1': 64,
    'layer1.2.act2': 64,
    'layer1.2.act3': 256,

    # === Layer 2 ===
    'layer2.0.act1': 128,
    'layer2.0.act2': 128,
    'layer2.0.act3': 512,

    'layer2.1.act1': 128,
    'layer2.1.act2': 128,
    'layer2.1.act3': 512,

    'layer2.2.act1': 128,
    'layer2.2.act2': 128,
    'layer2.2.act3': 512,

    'layer2.3.act1': 128,
    'layer2.3.act2': 128,
    'layer2.3.act3': 512,

    # === Layer 3 ===
    'layer3.0.act1': 256,
    'layer3.0.act2': 256,
    'layer3.0.act3': 1024,

    'layer3.1.act1': 256,
    'layer3.1.act2': 256,
    'layer3.1.act3': 1024,

    'layer3.2.act1': 256,
    'layer3.2.act2': 256,
    'layer3.2.act3': 1024,

    'layer3.3.act1': 256,
    'layer3.3.act2': 256,
    'layer3.3.act3': 1024,

    'layer3.4.act1': 256,
    'layer3.4.act2': 256,
    'layer3.4.act3': 1024,

    'layer3.5.act1': 256,
    'layer3.5.act2': 256,
    'layer3.5.act3': 1024,

    # === Layer 4 ===
    'layer4.0.act1': 512,
    'layer4.0.act2': 512,
    'layer4.0.act3': 2048,

    'layer4.1.act1': 512,
    'layer4.1.act2': 512,
    'layer4.1.act3': 2048,

    'layer4.2.act1': 512,
    'layer4.2.act2': 512,
    'layer4.2.act3': 2048,
}    

laplacian_gpaf_map_by_model = {
    name: {
        'type': LaplacianGPAF,
        'mode': 'shared',
        'shared_group': 'lap_all_shared'
    }
    for name in channel_map.keys()
}

all_3x3_shared_config = {
    name: {
        'type': LaplacianGPAF,
        'mode': 'shared',
        'shared_group': 'lap_3x3_shared'
    }
    for name in channel_map.keys() if name.endswith('act2')
}

laplacian_gpaf_map_by_channel = {
    'layer4.2.act2': {
        'type': LaplacianGPAF,
        'mode': 'channelwise'
    }
}

activations = {
    "full_relu":{},
    "laplacian_gpaf":laplacian_gpaf_map_by_channel,
    "laplacian_gpaf_by_model": laplacian_gpaf_map_by_model,
    "all_3x3_shared": all_3x3_shared_config,
}


class AllActivations:
    """Namespace for grouping all custom and standard activations used in the model."""
    ACTIVATION_TYPES = (
        CustomActivationPlaceholder,
        ChannelwiseActivation,
        LaplacianGPAF,
        KGActivation,
        nn.ReLU,
        # Add more if needed (e.g., nn.LeakyReLU, nn.Sigmoid)
    )