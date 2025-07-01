import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from utils.resnet_utils import build_activation_map, print_activation_map
from models.custom_activations import channel_map

from models.custom_activations import (
    KGActivationLaplacian,  # Formerly LaplacianGPAF
    KGActivationGeneral,    # Formerly KGActivation
    PReLUActivation,
    SwishFixed,
    SwishLearnable
)

# === Naming Convention for Activation Configs ===
# Format: <scope>_<positioning>_<sharing>_<activation_type>
#
# scope:
#   - stage1, stage4: specific ResNet stages (layer1, layer4)
#   - lastblock: only the final bottleneck block (layer4.2)
#   - laststage: all blocks in final stage (layer4)
#   - firstlast: first and last blocks
#   - all: all blocks in the network
#
# positioning:
#   - act2only: only the activation after the 3x3 convolution
#   - act123: all three activations in the bottleneck block
#
# sharing:
#   - shared: one shared parameter instance
#   - perlayer: different instance per layer (layer4, etc.)
#   - perblock: different instance per block (layer4.2, etc.)
#   - channelwise: different instance per channel
#
# activation_type:
#   - kglap: KGActivationLaplacian
#   - kggen: KGActivationGeneral
#   - prelu, swishfixed, swishlearn
#
# Example: 'laststage_act2only_channelwise_kglap'

def make_config(filter_fn, act_class, mode, group_fn=None):
    config = {}
    for name in channel_map:
        if filter_fn(name):
            entry = {'type': act_class, 'mode': mode}
            if mode == 'shared' and group_fn:
                entry['shared_group'] = group_fn(name)
            config[name] = entry
    return config

activations = {
    # === KGActivationLaplacian (kglap) ===

    # Use a single shared kglap activation for all activation points
    "all_act123_shared_kglap": make_config(
        lambda n: True, KGActivationLaplacian, "shared",
        lambda n: "kglap_all_shared"
    ),

    # Use separate shared kglap activations per stage and activation type (e.g., layer3_act2)
    "stage_act123_shared_kglap": make_config(
        lambda n: True, KGActivationLaplacian, "shared",
        lambda n: f"kglap_{n.split('.')[0]}_{n.split('.')[-1]}"
    ),

    # One shared kglap activation per stage, applied only to act2 (3x3) activations
    "stagegroup_act2only_shared_kglap": make_config(
        lambda n: n.endswith("act2"), KGActivationLaplacian, "shared",
        lambda n: f"kglap_stage_{n.split('.')[0]}_act2"
    ),

    # One shared kglap activation per block for act2 in stages 3 and 4
    "stage3_4_act2_blockshared_kglap": make_config(
        lambda n: (n.startswith("layer3") or n.startswith("layer4")) and n.endswith("act2"),
        KGActivationLaplacian, "shared",
        lambda n: f"kglap_{n.split('.')[0]}.{n.split('.')[1]}_act2"
    ),

    # Apply kglap to act2 in stage 4 only, each with its own parameters (per-channel)
    "stage4_act2only_channelwise_kglap": make_config(
        lambda n: n.startswith("layer4") and n.endswith("act2"), KGActivationLaplacian, "channelwise"
    ),

    # Use channelwise kglap activations for all act1, act2, and act3 points across the model
    "all_act123_channelwise_kglap": make_config(
        lambda n: True, KGActivationLaplacian, "channelwise"
    ),

    # === KGActivationGeneral (kggen) ===

    # Use a single shared kggen activation across the entire network
    "all_act123_shared_kggen": make_config(
        lambda n: True, KGActivationGeneral, "shared",
        lambda n: "kggen_all_shared"
    ),

    # Use separate shared kggen activations per stage and activation type
    "stage_act123_shared_kggen": make_config(
        lambda n: True, KGActivationGeneral, "shared",
        lambda n: f"kggen_{n.split('.')[0]}_{n.split('.')[-1]}"
    ),

    # Share a single kggen activation for all act1 and act3 points
    "act13_shared_kggen": make_config(
        lambda n: n.endswith("act1") or n.endswith("act3"), KGActivationGeneral, "shared",
        lambda n: "kggen_act13_shared"
    ),

    # Share a kggen activation per bottleneck block for act2 (e.g., layer3.4.act2)
    "block_act2_shared_kggen": make_config(
        lambda n: n.endswith("act2"), KGActivationGeneral, "shared",
        lambda n: f"kggen_{n.split('.')[0]}.{n.split('.')[1]}_act2"
    ),

    # Same as above but restricted to stages 3 and 4
    "stage3_4_act2_blockshared_kggen": make_config(
        lambda n: (n.startswith("layer3") or n.startswith("layer4")) and n.endswith("act2"),
        KGActivationGeneral, "shared",
        lambda n: f"kggen_{n.split('.')[0]}.{n.split('.')[1]}_act2"
    ),

    # Use channelwise kggen activations across all activation points
    "all_act123_channelwise_kggen": make_config(
        lambda n: True, KGActivationGeneral, "channelwise"
    ),

    # === Baseline variants ===

    # Channelwise PReLU (one learnable scalar per channel)
    "all_act123_channelwise_prelu": make_config(
        lambda n: True, lambda: PReLUActivation(num_parameters=1), "channelwise"
    ),

    # One shared SwishFixed activation for all positions
    "all_act123_shared_swishfixed": make_config(
        lambda n: True, SwishFixed, "shared", lambda n: "swish_fixed"
    ),

    # Channelwise SwishLearnable activation (each channel has its own beta)
    "all_act123_channelwise_swishlearn": make_config(
        lambda n: True, SwishLearnable, "channelwise"
    ),

    # All activations default to ReLU (used as a baseline)
    "full_relu": {}
}

def count_activation_params(activation_map):
    total_params = 0
    for act in activation_map.values():
        if hasattr(act, 'parameters'):
            total_params += sum(p.numel() for p in act.parameters() if p.requires_grad)
    return total_params


if __name__ == "__main__":
    print("Saving activation maps with parameter counts...")
    with open("activation_maps_output.txt", "w") as f:
        for config_name, config in activations.items():
            activation_map = build_activation_map(config) # Build the activation map using the config
            total = count_activation_params(activation_map)

            header = f"\n=== {config_name} ===\nTotal trainable activation parameters: {total}\n"
            print(header.strip())
            f.write(header)

            # Write each activation assignment
            for name, act in activation_map.items():
                act_type = type(act).__name__
                f.write(f"{name}: {act_type}\n")

                            
