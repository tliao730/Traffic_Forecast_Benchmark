import torch
import torch.nn as nn


def _ssm_fft_conv(x: torch.Tensor, lam: torch.Tensor,
                  B: torch.Tensor, C: torch.Tensor) -> torch.Tensor:
    """
    LTI SSM via FFT convolution (real-valued).
        h_t = lam * h_{t-1} + B @ x_t
        y_t = C @ h_t

    x:   (B, L, D)
    lam: (D, S)   values in (0, 1)
    B:   (S, D)   weight matrix of self.B
    C:   (D, S)   weight matrix of self.C
    Returns y: (B, L, D)
    """
    _, L, _ = x.shape

    t = torch.arange(L, device=lam.device, dtype=lam.dtype)
    lam_t = lam.unsqueeze(-1) ** t.unsqueeze(0).unsqueeze(0)  # (D, S, L)

    Bx = x @ B.T  # (B, L, S)

    fft_len = 1 << (2 * L - 1).bit_length()
    # kernel per state: lam_t is (D, S, L), need (S, D, L) for conv with (B, L, S)
    # For each state s: y_s[b, t, d] = sum_tau C[d,s]*lam[d,s]^tau * Bx[b, t-tau, s]
    # Merge: y = sum_s (C[:,s] * lam[:,s]^tau) conv Bx[:,:,s]
    # kernel_s: (D, L) for state s = C[:,s:s+1] * lam[:,s:s+1]^t -> (D, L)

    # FFT over Bx per state and kernel per state
    Bx_f = torch.fft.rfft(Bx.permute(0, 2, 1), n=fft_len)  # (B, S, fft_len//2+1)
    k_f  = torch.fft.rfft(lam_t, n=fft_len)                 # (D, S, fft_len//2+1)

    # y[b, d, t] = sum_s C[d,s] * ifft(k_f[d,s] * Bx_f[b,s])
    # broadcast: k_f (D,S,F) * Bx_f (B,S,F) -> need (B,D,S,F)
    conv = k_f.unsqueeze(0) * Bx_f.unsqueeze(1)  # (B, D, S, F)
    y_s  = torch.fft.irfft(conv, n=fft_len)[..., :L]  # (B, D, S, L)
    # weight by C and sum over states
    y = (C.unsqueeze(0).unsqueeze(-1) * y_s).sum(dim=2)  # (B, D, L)
    return y.permute(0, 2, 1)  # (B, L, D)


class LinearSSMLayer(nn.Module):
    """
    Minimal diagonal LTI SSM solved via FFT convolution:
        h_t = diag(lambda) * h_{t-1} + B * x_t
        y_t = C * h_t + D * x_t

    lambda constrained to (0, 1) via sigmoid for stability.
    """

    def __init__(self, d_model: int, d_state: int = 16):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.log_lambda = nn.Parameter(torch.zeros(d_model, d_state))
        self.B = nn.Linear(d_model, d_state, bias=False)
        self.C = nn.Linear(d_state, d_model, bias=False)
        self.D = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lam = torch.sigmoid(self.log_lambda)  # (D, S)
        y = _ssm_fft_conv(x, lam, self.B.weight, self.C.weight)
        return y + self.D * x


class LinearSSMForecastModel(nn.Module):
    """
    Base model: stack of minimal diagonal LTI SSM layers with pre-norm and residual.
    Input:  (B, context_length, 1)
    Output: (B, prediction_length)
    """

    def __init__(self, prediction_length: int, d_model: int = 64,
                 d_state: int = 16, num_layers: int = 2):
        super().__init__()
        self.input_proj = nn.Linear(1, d_model)
        self.norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(num_layers)])
        self.layers = nn.ModuleList([LinearSSMLayer(d_model, d_state) for _ in range(num_layers)])
        self.out_proj = nn.Linear(d_model, prediction_length)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(x)
        for norm, layer in zip(self.norms, self.layers):
            h = layer(norm(h)) + h
        return self.out_proj(h[:, -1, :])

    def param_num(self) -> int:
        return sum(p.numel() for p in self.parameters())
