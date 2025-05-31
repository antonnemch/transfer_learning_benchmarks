from torch.optim.sgd import SGD

class MetaSGD(SGD):
    def __init__(self, net, *args, **kwargs):
        super(MetaSGD, self).__init__(*args, **kwargs)
        self.net = net
        # Build a list that assigns each parameter to a group index (used to select its learning rate)
        self.indices = self._build_indices()

    def _build_indices(self):
        """
        Dynamically assign learning rate group indices to each parameter based on its name.

        Group mapping:
            0: conv1
            1: bn1
            2–4: layer1[0–2]
            5–8: layer2[0–3]
            9–14: layer3[0–5]
            15–17: layer4[0–2]
            18: fc
        """
        name_to_group = {}
        for i in range(3):  # 3 Bottleneck blocks in layer1
            name_to_group[f'layer1.{i}.'] = 2 + i
        for i in range(4):  # 4 Bottlenecks in layer2
            name_to_group[f'layer2.{i}.'] = 5 + i
        for i in range(6):  # 6 Bottlenecks in layer3
            name_to_group[f'layer3.{i}.'] = 9 + i
        for i in range(3):  # 3 Bottlenecks in layer4
            name_to_group[f'layer4.{i}.'] = 15 + i

        indices = []
        for name, _ in self.net.named_parameters():
            if name.startswith("conv1."):
                indices.append(0)  # Initial convolution
            elif name.startswith("bn1."):
                indices.append(1)  # Initial batch norm
            elif name.startswith("fc."):
                indices.append(18)  # Final classification head
            else:
                # Match the parameter name to its corresponding group index
                matched = False
                for prefix, idx in name_to_group.items():
                    if name.startswith(prefix):
                        indices.append(idx)
                        matched = True
                        break
                if not matched:
                    raise ValueError(f"Unmatched parameter name in MetaSGD: {name}")
        return indices

    def set_parameter(self, current_module, name, parameters):
        """
        Recursively set a parameter in the model. Used to apply updated weights.
        """
        if '.' in name:
            name_split = name.split('.')
            module_name = name_split[0]
            rest_name = '.'.join(name_split[1:])
            for child_name, child in current_module.named_children():
                if module_name == child_name:
                    self.set_parameter(child, rest_name, parameters)
                    break
        else:
            current_module._parameters[name] = parameters

    def meta_step(self, grads, lrs):
        """
        Apply a meta-learning gradient step: update each parameter using its
        corresponding learning rate group (lrs[group_id]).

        Arguments:
            grads: List of gradients for each parameter (same order as named_parameters()).
            lrs: List of group-level learning rates (length 19 for ResNet-50).
        """
        group = self.param_groups[0]
        weight_decay = group['weight_decay']
        momentum = group['momentum']
        dampening = group['dampening']
        nesterov = group['nesterov']

        for i, ((name, param), grad) in enumerate(zip(self.net.named_parameters(), grads)):
            param.detach_()  # Detach from the computation graph

            # Apply weight decay if specified
            if weight_decay != 0:
                grad = grad + weight_decay * param

            # Apply momentum if available
            if momentum != 0 and 'momentum_buffer' in self.state[param]:
                buf = self.state[param]['momentum_buffer']
                grad = buf.mul(momentum).add(grad, alpha=1 - dampening)

            # Optional: Nesterov accelerated gradient
            if nesterov:
                grad = grad + momentum * grad

            # Determine which group this parameter belongs to
            group_idx = self.indices[i]

            # Perform the meta-learning update: θ ← θ - η * ∇
            update = lrs[group_idx] * grad
            self.set_parameter(self.net, name, param - update)
