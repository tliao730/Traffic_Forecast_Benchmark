import torch
import torch.nn as nn
import torch.nn.functional as F


class SelectiveSSM(nn.Module):
    """S6: input-dependent B, C, dt. Sequential scan."""

    def __init__(self, d_inner: int, d_state: int = 16):
        super().__init__()
        self.d_inner = d_inner
        self.d_state = d_state
        A = torch.arange(1, d_state + 1).float().unsqueeze(0).expand(d_inner, -1)
        self.A_log  = nn.Parameter(torch.log(A))
        self.D      = nn.Parameter(torch.ones(d_inner))
        self.dt_proj = nn.Linear(d_inner, d_inner)
        self.dt_bias = nn.Parameter(torch.zeros(d_inner))
        self.B_proj  = nn.Linear(d_inner, d_state, bias=False)
        self.C_proj  = nn.Linear(d_inner, d_state, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, d_inner)
        B, L, D = x.shape
        A     = -torch.exp(self.A_log.float())              # (D, S)
        dt    = F.softplus(self.dt_proj(x) + self.dt_bias)  # (B, L, D)
        B_seq = self.B_proj(x)                              # (B, L, S)
        C_seq = self.C_proj(x)                              # (B, L, S)

        h = x.new_zeros(B, D, self.d_state)
        ys = []
        for t in range(L):
            dt_t  = dt[:, t, :, None]
            A_bar = torch.exp(dt_t * A.unsqueeze(0))
            B_bar = dt_t * B_seq[:, t, None, :]
            h = A_bar * h + B_bar * x[:, t, :, None]
            ys.append((C_seq[:, t, None, :] * h).sum(-1))
        return torch.stack(ys, dim=1) + self.D * x          # (B, L, D)


class MambaBlock(nn.Module):
    """Mamba block: expand → conv1d → S6 → gate → residual."""

    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4, expand: int = 2):
        super().__init__()
        d_inner = d_model * expand
        self.norm    = nn.LayerNorm(d_model)
        self.in_proj = nn.Linear(d_model, d_inner * 2, bias=False)
        self.conv1d  = nn.Conv1d(d_inner, d_inner, d_conv,
                                 padding=d_conv - 1, groups=d_inner)
        self.ssm     = SelectiveSSM(d_inner, d_state)
        self.out_proj = nn.Linear(d_inner, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        x_in, z = self.in_proj(x).chunk(2, dim=-1)
        x_in = self.conv1d(x_in.transpose(1, 2))[..., :residual.shape[1]]
        x_in = F.silu(x_in.transpose(1, 2))
        y = self.ssm(x_in) * F.silu(z)
        return self.out_proj(y) + residual


class MambaForecastModel(nn.Module):
    """
    Univariate Mamba — no spatial head.
    Each sensor is an independent time series.
    Input:  (B, context_length, 1)
    Output: (B, prediction_length)
    """

    def __init__(self, prediction_length: int, d_model: int = 64,
                 d_state: int = 16, d_conv: int = 4,
                 expand: int = 2, num_layers: int = 2):
        super().__init__()
        self.input_proj = nn.Linear(1, d_model)
        self.blocks = nn.ModuleList([
            MambaBlock(d_model, d_state, d_conv, expand)
            for _ in range(num_layers)
        ])
        self.out_proj = nn.Linear(d_model, prediction_length)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, 1)
        h = self.input_proj(x)      # (B, T, d_model)
        for block in self.blocks:
            h = block(h)
        return self.out_proj(h[:, -1, :])  # (B, prediction_length)

    def param_num(self) -> int:
        return sum(p.numel() for p in self.parameters())
