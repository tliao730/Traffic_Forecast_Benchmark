import torch
import torch.nn as nn
import torch.nn.functional as F


def parallel_scan(A_bar: torch.Tensor, Bx: torch.Tensor) -> torch.Tensor:
    """
    Parallel associative scan for the linear recurrence:
        h_t = A_t * h_{t-1} + Bx_t

    Uses the identity that (a2, b2) ∘ (a1, b1) = (a2*a1, a2*b1 + b2),
    which lets us lift the sequential recurrence into a tree of parallel
    reductions — O(log L) passes instead of O(L) sequential steps.

    Args:
        A_bar: (B, L, D, S)  — discretised decay per step
        Bx:   (B, L, D, S)  — discretised input contribution per step

    Returns:
        h:    (B, L, D, S)  — hidden state at every time step
    """
    B, L, D, S = A_bar.shape

    # Pad L up to the next power of two so the tree is balanced.
    L_pad = 1 << (L - 1).bit_length()
    if L_pad > L:
        pad = L_pad - L
        # Neutral element for the monoid: (1, 0) — identity under ∘
        A_bar = F.pad(A_bar, (0, 0, 0, 0, 0, pad), value=1.0)
        Bx    = F.pad(Bx,    (0, 0, 0, 0, 0, pad), value=0.0)

    a = A_bar  # (B, L_pad, D, S)
    b = Bx     # (B, L_pad, D, S)

    # ── Up-sweep (reduce) ─────────────────────────────────────────────────
    # After this phase a[:, i] holds the combined (A, Bx) over the subtree
    # rooted at position i.
    levels = []
    cur_len = L_pad
    while cur_len > 1:
        cur_len //= 2
        a_left  = a[:, :cur_len]           # (B, cur_len, D, S)
        a_right = a[:, cur_len:2*cur_len]
        b_left  = b[:, :cur_len]
        b_right = b[:, cur_len:2*cur_len]
        levels.append((a_left.clone(), b_left.clone()))
        # Combine pairs: right ∘ left
        b = a_right * b_left + b_right
        a = a_right * a_left

    # ── Down-sweep (scan) ─────────────────────────────────────────────────
    # Propagate prefix products back down the tree.
    h_acc_a = torch.ones_like(a[:, :1])   # accumulated A  (identity = 1)
    h_acc_b = torch.zeros_like(b[:, :1])  # accumulated Bx (identity = 0)

    for a_left, b_left in reversed(levels):
        # Expand accumulators to match this level's width
        h_acc_a = h_acc_a.repeat(1, 2, 1, 1)
        h_acc_b = h_acc_b.repeat(1, 2, 1, 1)
        half = h_acc_a.shape[1] // 2
        # Left child: accumulator unchanged
        # Right child: combine accumulator with left sibling
        new_a_right = h_acc_a[:, :half] * a_left
        new_b_right = h_acc_a[:, :half] * b_left + h_acc_b[:, :half]
        h_acc_a = torch.cat([h_acc_a[:, :half], new_a_right], dim=1)
        h_acc_b = torch.cat([h_acc_b[:, :half], new_b_right], dim=1)

    # h_acc_b now holds the prefix-sum h at every position; trim padding.
    return h_acc_b[:, :L]  # (B, L, D, S)


class SelectiveSSMParallel(nn.Module):
    """S6 with GPU parallel associative scan instead of sequential for-loop."""

    def __init__(self, d_inner: int, d_state: int = 16):
        super().__init__()
        self.d_inner = d_inner
        self.d_state = d_state
        A = torch.arange(1, d_state + 1).float().unsqueeze(0).expand(d_inner, -1)
        self.A_log   = nn.Parameter(torch.log(A))
        self.D       = nn.Parameter(torch.ones(d_inner))
        self.dt_proj = nn.Linear(d_inner, d_inner)
        self.dt_bias = nn.Parameter(torch.zeros(d_inner))
        self.B_proj  = nn.Linear(d_inner, d_state, bias=False)
        self.C_proj  = nn.Linear(d_inner, d_state, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, d_inner)
        B, L, D = x.shape
        S = self.d_state

        A     = -torch.exp(self.A_log.float())               # (D, S)
        dt    = F.softplus(self.dt_proj(x) + self.dt_bias)   # (B, L, D)
        B_seq = self.B_proj(x)                               # (B, L, S)
        C_seq = self.C_proj(x)                               # (B, L, S)

        # Discretise: ZOH (zero-order hold)
        # dt: (B,L,D) → (B,L,D,1);  A: (D,S) → (1,1,D,S)
        dt4  = dt.unsqueeze(-1)
        A_bar = torch.exp(dt4 * A.unsqueeze(0).unsqueeze(0))  # (B, L, D, S)

        # Bx = dt * B_seq * x  — broadcast carefully
        # dt: (B,L,D), x: (B,L,D), B_seq: (B,L,S)
        # want Bx: (B, L, D, S)
        Bx = dt4 * (x.unsqueeze(-1) * B_seq.unsqueeze(2))     # (B, L, D, S)

        # Parallel scan → h at every time step: (B, L, D, S)
        h = parallel_scan(A_bar, Bx)

        # Read out: y_t = C_t · h_t  → sum over S
        # C_seq: (B,L,S) → (B,L,1,S); h: (B,L,D,S)
        y = (C_seq.unsqueeze(2) * h).sum(-1)                   # (B, L, D)

        return y + self.D * x                                  # (B, L, D)


class MambaBlockV2(nn.Module):
    """Mamba block using parallel SSM scan."""

    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4, expand: int = 2):
        super().__init__()
        # Expand inner dimension to give the block more expressive capacity.
        d_inner = d_model * expand
        # Normalise input before processing to stabilise training.
        self.norm     = nn.LayerNorm(d_model)
        # Project to 2x d_inner so we can split into SSM path and gate path.
        self.in_proj  = nn.Linear(d_model, d_inner * 2, bias=False)
        # Depthwise causal conv: each channel attends independently to its local history.
        self.conv1d   = nn.Conv1d(d_inner, d_inner, d_conv,
                                  padding=d_conv - 1, groups=d_inner)
        # Selective SSM: captures long-range dependencies via learned recurrence.
        self.ssm      = SelectiveSSMParallel(d_inner, d_state)
        # Project back to d_model so the block is shape-preserving.
        self.out_proj = nn.Linear(d_inner, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Save input for the residual connection at the end.
        residual = x
        x = self.norm(x)

        # Split into SSM path (x_in) and multiplicative gate (z).
        x_in, z = self.in_proj(x).chunk(2, dim=-1)

        # Conv1d expects (B, D, L); transpose in, apply conv, transpose back.
        # Trim the extra timesteps introduced by causal padding.
        x_in = self.conv1d(x_in.transpose(1, 2))[..., :residual.shape[1]]
        x_in = F.silu(x_in.transpose(1, 2))

        # SSM output gated by z: controls how much of the memory to pass through.
        y = self.ssm(x_in) * F.silu(z)

        # Project back to d_model and add residual to ease gradient flow.
        return self.out_proj(y) + residual


class MambaForecastModelV2(nn.Module):
    """
    Univariate Mamba with parallel SSM scan.
    Input:  (B, context_length, 1)
    Output: (B, prediction_length)
    """

    def __init__(self, prediction_length: int, d_model: int = 64,
                 d_state: int = 16, d_conv: int = 4,
                 expand: int = 2, num_layers: int = 2):
        super().__init__()
        self.input_proj = nn.Linear(1, d_model)
        self.blocks = nn.ModuleList([
            MambaBlockV2(d_model, d_state, d_conv, expand)
            for _ in range(num_layers)
        ])
        self.out_proj = nn.Linear(d_model, prediction_length)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(x)
        for block in self.blocks:
            h = block(h)
        return self.out_proj(h[:, -1, :])

    def param_num(self) -> int:
        return sum(p.numel() for p in self.parameters())
