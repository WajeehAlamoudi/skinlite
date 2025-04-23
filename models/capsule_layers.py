import torch
import torch.nn as nn
import torch.nn.functional as F

import config

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1, 1280, 7, 7
class PrimaryCaps(nn.Module):
    def __init__(self, num_capsules=8, in_channels=1280, out_channels=32):
        super(PrimaryCaps, self).__init__()
        self.capsules = nn.ModuleList([
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
            for _ in range(num_capsules)
        ])

    def forward(self, x):
        batch_size = x.size(0)
        u = [capsule(x).view(batch_size, 32 * 7 * 7, 1) for capsule in self.capsules]
        u = torch.cat(u, dim=-1)
        u_squashed = self.squash(u)
        return u_squashed

    def squash(self, x):
        squared_norm = (x ** 2).sum(dim=-1, keepdim=True)
        scale = squared_norm / (1 + squared_norm)
        output = scale * x / torch.sqrt(squared_norm)
        return output


def softmax(x, dim=1):
    transposed_inp = x.transpose(dim, len(x.size()) - 1)
    softmaxed = F.softmax(transposed_inp.contiguous().view(-1, transposed_inp.size(-1)), dim=-1)
    return softmaxed.view(*transposed_inp.size()).transpose(dim, len(x.size()) - 1)


def dynamic_routing(b_ij, u_hat, squash, routing_iterations=3):
    for iterations in range(routing_iterations):
        c_ij = softmax(b_ij, dim=2)
        s_j = (c_ij * u_hat).sum(dim=2, keepdim=True)
        v_j = squash(s_j)
        if iterations < routing_iterations - 1:
            a_ij = (u_hat * v_j).sum(dim=-1, keepdim=True)
            b_ij = b_ij + a_ij
    return v_j


class DigitCaps(nn.Module):
    def __init__(self, num_caps=7, previous_layer_nodes=32 * 7 * 7,
                 in_channels=8, out_channels=16):
        super(DigitCaps, self).__init__()
        self.num_caps = num_caps
        self.previous_layer_nodes = previous_layer_nodes
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.W = nn.Parameter(torch.randn(num_caps, previous_layer_nodes,
                                          in_channels, out_channels))

    def forward(self, x):
        batch_size = x.size(0)  # e.g., 1
        input_caps = x.size(1)  # e.g., 1568
        x = x.unsqueeze(1).unsqueeze(3)  # [B, 1, input_caps, 1, in_channels] → [1, 1, 1568, 1, 8]

        W = self.W.unsqueeze(0)  # [1, num_caps, input_caps, in_channels, out_channels]
        W = W.expand(batch_size, -1, -1, -1, -1)

        x = x.expand(-1, self.num_caps, -1, -1, -1)  # match W: [B, num_caps, input_caps, 1, in_channels]

        # Now: matmul along last two dims: [1, 7, 1568, 1, 8] @ [1, 7, 1568, 8, 16] → [1, 7, 1568, 1, 16]
        u_hat = torch.matmul(x, W)  # shape: [B, 7, 1568, 1, 16]

        b_ij = torch.zeros_like(u_hat)

        if x.device.type != 'cpu':
            b_ij = b_ij.to(x.device)

        v_j = dynamic_routing(b_ij, u_hat, self.squash, routing_iterations=3)

        v_j = v_j.squeeze(3)  # Remove the singleton dim → [B, 7, 16]
        return v_j

    def squash(self, x):
        squared_norm = (x ** 2).sum(dim=-1, keepdim=True)
        scale = squared_norm / (1 + squared_norm)
        out = scale * x / torch.sqrt(squared_norm)
        return out


class Decoder(nn.Module):
    def __init__(self, input_vector_length=16, input_capsules=16, hidden_dim=512):
        super(Decoder, self).__init__()
        input_dim = input_vector_length*input_capsules
        self.lin_layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim*2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim*2, 3*config.run_config['IMAGE_SIZE']*config.run_config['IMAGE_SIZE']),
            nn.Sigmoid()
        )

    def forward(self, x, y):
        if x.dim() == 4 and x.size(2) == 1:
            x = x.squeeze(2)

        batch_size, num_classes, capsule_dim = x.size()

        # Mask only the desired class capsule (y should be one-hot)
        x = x * y[:, :, None]

        # Flatten for MLP
        flattened_x = x.view(batch_size, num_classes * capsule_dim)

        # Pass through reconstruction layers
        reconstructed = self.lin_layers(flattened_x)

        return reconstructed





