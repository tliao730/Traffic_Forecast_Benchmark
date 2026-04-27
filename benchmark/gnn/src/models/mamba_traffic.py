import torch
import torch.nn as nn
import torch.nn.functional as F
from gnn.src.base.model import BaseModel


class SelectiveSSM(nn.Module):
    """S6: B, C, dt are input-dependent (selective mechanism). Sequential scan."""

    def __init__(self, d_inner: int, d_state: int = 16):
        super().__init__()
        self.d_inner = d_inner
        self.d_state = d_state

        A = torch.arange(1, d_state + 1, dtype=torch.float32).unsqueeze(0).expand(d_inner, -1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(d_inner))

        self.dt_proj = nn.Linear(d_inner, d_inner)
        self.dt_bias = nn.Parameter(torch.zeros(d_inner))
        self.B_proj = nn.Linear(d_inner, d_state, bias=False)
        self.C_proj = nn.Linear(d_inner, d_state, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, d_inner)
        A = -torch.exp(self.A_log.float())                 # (d_inner, d_state)
        dt = F.softplus(self.dt_proj(x) + self.dt_bias)   # (B, L, d_inner)
        B_seq = self.B_proj(x)                             # (B, L, d_state)
        C_seq = self.C_proj(x)                             # (B, L, d_state)

        B, L, D = x.shape
        h = x.new_zeros(B, D, self.d_state)
        ys = []
        for t in range(L):
            dt_t = dt[:, t, :, None]                       # (B, D, 1)
            A_bar = torch.exp(dt_t * A.unsqueeze(0))       # (B, D, S)
            B_bar = dt_t * B_seq[:, t, None, :]            # (B, D, S)
            h = A_bar * h + B_bar * x[:, t, :, None]
            y = (C_seq[:, t, None, :] * h).sum(-1)         # (B, D)
            ys.append(y)

        return torch.stack(ys, dim=1) + self.D * x         # (B, L, d_inner)


class MambaBlock(nn.Module):
    """Mamba block: expand → causal conv1d → S6 → gated output → residual."""

    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4, expand: int = 2):
        super().__init__()
        d_inner = d_model * expand

        self.norm = nn.LayerNorm(d_model)
        self.in_proj = nn.Linear(d_model, d_inner * 2, bias=False)
        self.conv1d = nn.Conv1d(
            d_inner, d_inner, kernel_size=d_conv,
            padding=d_conv - 1, groups=d_inner, bias=True,
        )
        self.ssm = SelectiveSSM(d_inner, d_state)
        self.out_proj = nn.Linear(d_inner, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, d_model)
        residual = x
        x = self.norm(x)
        x_in, z = self.in_proj(x).chunk(2, dim=-1)         # each: (B, L, d_inner)

        x_in = self.conv1d(x_in.transpose(1, 2))[..., :residual.shape[1]]
        x_in = F.silu(x_in.transpose(1, 2))

        y = self.ssm(x_in) * F.silu(z)
        return self.out_proj(y) + residual


class TrafficMambaBackbone(nn.Module):
    """
    Node-agnostic temporal encoder.
    Input:  (B, T, input_dim)
    Output: (B, T, d_model)
    """

    def __init__(self, input_dim: int, d_model: int, d_state: int,
                 d_conv: int, expand: int, num_layers: int):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.blocks = nn.ModuleList([
            MambaBlock(d_model, d_state, d_conv, expand) for _ in range(num_layers)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        for block in self.blocks:
            x = block(x)
        return x


class TrafficMamba(BaseModel):
    """
    Mamba for traffic forecasting.
    Backbone processes each sensor independently; spatial head mixes across sensors.
    """

    def __init__(self, d_model: int = 64, d_state: int = 16, d_conv: int = 4,
                 expand: int = 2, num_layers: int = 2, embed_dim: int = 10, **args):
        super().__init__(**args)
        self.backbone = TrafficMambaBackbone(
            self.input_dim, d_model, d_state, d_conv, expand, num_layers
        )
        self.node_embed = nn.Parameter(torch.randn(self.node_num, embed_dim))
        self.spatial_fc = nn.Linear(d_model + embed_dim, d_model)
        self.out_proj = nn.Conv2d(
            1, self.horizon * self.output_dim,
            kernel_size=(1, d_model), bias=True,
        )

    def forward(self, source: torch.Tensor, label=None) -> torch.Tensor:
        # source: (B, T, N, F_in)
        B, T, N, F_in = source.shape

        # treat each sensor as an independent sequence
        x = source.permute(0, 2, 1, 3).reshape(B * N, T, F_in)  # (B*N, T, F_in)
        x = self.backbone(x)                                    # (B*N, T, d_model)
        x = x[:, -1, :].reshape(B, N, -1)                      # (B, N, d_model)

        # spatial mixing
        emb = self.node_embed.unsqueeze(0).expand(B, -1, -1)   # (B, N, embed_dim)
        x = F.relu(self.spatial_fc(torch.cat([x, emb], dim=-1)))  # (B, N, d_model)

        x = x.unsqueeze(1)                                      # (B, 1, N, d_model)
        return self.out_proj(x)                                 # (B, H, N, 1)

    def load_pretrained_backbone(self, ckpt_path: str, device: str = "cpu"):
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
        backbone_state = {
            k[len("backbone."):]: v
            for k, v in ckpt.items() if k.startswith("backbone.")
        }
        return self.backbone.load_state_dict(backbone_state, strict=True)
