import torch
import torch.nn as nn
import torch.nn.functional as F


class CapsuleLayer(nn.Module):
    def __init__(self, in_units, in_channels, num_units, unit_size, use_routing):
        super(CapsuleLayer, self).__init__()

        self.in_units = in_units              # used only for routing
        self.in_channels = in_channels        # input capsule count or CNN channels
        self.num_units = num_units            # output capsule count
        self.unit_size = unit_size            # output capsule dimension
        self.use_routing = use_routing

        if use_routing:
            # Routing layer: learn transformation matrix
            self.W = nn.Parameter(torch.randn(1, in_channels, num_units, in_units, unit_size))
        else:
            # PrimaryCaps: apply a Conv2D per capsule unit
            self.units = nn.ModuleList([
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=unit_size,
                    kernel_size=5,
                    stride=1,
                    padding=0
                ) for _ in range(num_units)
            ])

    @staticmethod
    def squash(s):
        mag_sq = torch.sum(s**2, dim=2, keepdim=True)
        mag = torch.sqrt(mag_sq + 1e-8)
        return (mag_sq / (1.0 + mag_sq)) * (s / (mag + 1e-8))

    def forward(self, x):
        return self.routing(x) if self.use_routing else self.no_routing(x)

    def no_routing(self, x):
        u = [self.units[i](x) for i in range(self.num_units)]          # list of [B, unit_size, H, W]
        u = torch.stack(u, dim=1)                                      # [B, num_units, unit_size, H, W]
        # print(f"▶️ PrimaryCaps output shape before flatten: {u.shape}")
        u = u.view(x.size(0), self.num_units, -1)                      # [B, num_units, unit_size * H * W]
        # print(f"✅ PrimaryCaps output shape: {u.shape}")
        return CapsuleLayer.squash(u)                                  # [B, num_units, output_dim]

    def routing(self, x):
        # print("Start routing Digital")

        B, C, I = x.shape  # [B, in_channels, in_units]
        x = x.unsqueeze(2).unsqueeze(4)  # [B, C, 1, I, 1]
        x = x.permute(0, 1, 2, 4, 3)  # [B, C, 1, 1, I]
        # print("x reshaped:", x.shape)

        W = self.W.expand(B, -1, -1, -1, -1)  # [B, C, U, I, D]
        # print("W shape:", W.shape)

        u_hat = torch.matmul(x, W)  # [B, C, U, 1, D]
        u_hat = u_hat.permute(0, 1, 2, 4, 3)  # [B, C, U, D, 1]
        # print("u_hat shape:", u_hat.shape)

        b_ij = torch.zeros(B, C, self.num_units, 1, device=x.device)

        for _ in range(3):
            c_ij = F.softmax(b_ij, dim=2).unsqueeze(-1)  # [B, C, U, 1, 1]
            s_j = (c_ij * u_hat).sum(dim=1, keepdim=True)  # [B, 1, U, D, 1]
            v_j = CapsuleLayer.squash(s_j)

            if _ < 2:
                u_vj = torch.matmul(u_hat.transpose(3, 4), v_j).squeeze(-1)  # [B, C, U, 1]
                b_ij = b_ij + u_vj

        return v_j.squeeze(1).squeeze(-1)  # [B, U, D]

