import torch
from models.resnet_base import initialize_basic_model
from models.activation_configs import activations
from utils.resnet_utils import build_activation_map

# Choose a config and number of classes for the test
activation_type = "stage4.2_act123_channelwise_kglap"
num_classes = 2

def print_bn_status(model):
    print("BatchNorm requires_grad status:")
    for name, module in model.named_modules():
        if "bn" in name.lower() and hasattr(module, "weight"):
            print(f"  {name}: weight requires_grad={module.weight.requires_grad}, "
                  f"bias requires_grad={getattr(module.bias, 'requires_grad', None)}")

for train_bn in [False, True]:
    print(f"\n=== Testing with TrainBN={train_bn} ===")
    model = initialize_basic_model(num_classes, device="cpu", freeze=True)
    activation_map = build_activation_map(activations[activation_type])
    model.set_custom_activation_map(activation_map, train_bn=train_bn)
    print_bn_status(model)
