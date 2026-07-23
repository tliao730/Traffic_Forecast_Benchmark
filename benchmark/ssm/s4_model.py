import math
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# HiPPO-LegS initialisation helpers
# ---------------------------------------------------------------------------

def _hippo_legs_A(N: int) -> torch.Tensor:
    """
    HiPPO-LegS state matrix A of size (N, N).
    A_{nk} = -sqrt(2n+1)*sqrt(2k+1)  if n > k
           = -(n+1)                   if n == k
           = 0                        if n < k
    """
    n = torch.arange(N, dtype=torch.float64)
    k = torch.arange(N, dtype=torch.float64)
    A = -torch.sqrt(2 * n[:, None] + 1) * torch.sqrt(2 * k[None, :] + 1)
    A = torch.tril(A)
    A.diagonal().copy_(-(n + 1))
    return A.float()


def _nplr_hippo(N: int):
    """
    Decompose HiPPO-LegS into NPLR form: A = V diag(Λ) V^{-1} - p q^*
    Returns (Λ, p, q, V) all complex, size (N,) or (N,).
    Λ: eigenvalues, p/q: low-rank vectors, V: eigenvector matrix.
    """
    A = _hippo_legs_A(N).double()

    # Skew-symmetric + symmetric decomposition: A = S + K
    # S = (A - A^T)/2  (skew), K = (A + A^T)/2  (symmetric, rank-1 for HiPPO)
    S = (A - A.T) / 2
    K = (A + A.T) / 2

    # K for HiPPO-LegS is rank-1: K = -p p^T where p_n = sqrt((2n+1)/2) / sqrt(2)
    # Recover p from the diagonal of -K
    p = torch.sqrt(-K.diagonal())              # (N,) real, rank-1 vector

    # Diagonalise the skew-Hermitian matrix S (purely imaginary eigenvalues)
    # iS is symmetric real → standard eigh
    eigvals, V = torch.linalg.eigh(S * 1j)    # eigvals real (imag part of iS eig)
    Lambda = eigvals / 1j                      # purely imaginary: (N,) complex

    # p and q in the eigenbasis
    Vc = V.to(torch.complex128)
    p_c = p.to(torch.complex128)
    p_eig = Vc.conj().T @ p_c                 # (N,) complex
    q_eig = Vc.conj().T @ p_c                 # same as p for HiPPO (p == q)

    return Lambda.cfloat(), p_eig.cfloat(), q_eig.cfloat(), Vc.cfloat()


# ---------------------------------------------------------------------------
# Cauchy kernel  k[t] = C (zI - A)^{-1} B  evaluated on the unit circle
#
# VECTORISED over the channel dimension D, so a single call handles all
# d_model channels at once instead of looping in Python.
# ---------------------------------------------------------------------------

def _cauchy_dot_batched(v: torch.Tensor, z: torch.Tensor, lam: torch.Tensor) -> torch.Tensor:
    """
    Compute, for every channel d independently:
        out[d, f] = sum_n v[d, n] / (z[f] - lam[d, n])

    v, lam: (D, N) complex   — per-channel vectors
    z:      (L,) complex     — shared frequency grid (roots of unity), same
                               for every channel since L is common
    Returns: (D, L) complex
    """
    # Broadcast: (D, 1, N) vs (1, L, 1) -> (D, L, N), then sum over N
    diff = z.view(1, -1, 1) - lam.unsqueeze(1)          # (D, L, N)
    return (v.unsqueeze(1) / diff).sum(-1)               # (D, L)


def _s4_kernel_batched(Lambda: torch.Tensor, p: torch.Tensor, q: torch.Tensor,
                        B: torch.Tensor, C: torch.Tensor,
                        log_dt: torch.Tensor, L: int) -> torch.Tensor:
    """
    Compute the S4 convolution kernel k ∈ R^{D x L} for ALL d_model channels
    at once via the Woodbury identity, replacing the previous per-channel
    Python for-loop with batched tensor ops.

        (zI - A)^{-1} = (zI - diag(Λ) + pq^*)^{-1}
                      = (zI - Λ)^{-1} + (zI-Λ)^{-1} p [1 - q^*(zI-Λ)^{-1}p]^{-1} q^*(zI-Λ)^{-1}

    Args:
        Lambda, p, q, B, C: (D, N) complex — per-channel NPLR components
        log_dt: (D,) — per-channel log step size
        L: sequence length

    Returns:
        k: (D, L) real
    """
    device = Lambda.device
    dt = torch.exp(log_dt)                                          # (D,)

    # Each channel can have its own dt, so the s-domain sample grid is
    # channel-specific even though omega (roots of unity) is shared.
    omega = torch.exp(-2j * math.pi * torch.arange(L, device=device) / L)  # (L,)
    # s = 2/dt * (omega-1)/(omega+1), per channel: (D, L)
    s = (2 / dt).unsqueeze(-1) * ((omega - 1) / (omega + 1)).unsqueeze(0)  # (D, L)

    # Cauchy terms, computed per-channel since s is channel-dependent.
    # We still avoid the Python loop over d_model by using a batched
    # Cauchy dot that takes a *per-channel* z grid.
    def cauchy_per_channel(v, z_per_chan, lam):
        # v, lam: (D, N); z_per_chan: (D, L) -> diff: (D, L, N)
        diff = z_per_chan.unsqueeze(-1) - lam.unsqueeze(1)   # (D, L, N)
        return (v.unsqueeze(1) / diff).sum(-1)                # (D, L)

    cb = cauchy_per_channel(B, s, Lambda)                     # (D, L)
    cq = cauchy_per_channel(q.conj(), s, Lambda)               # (D, L)
    cp = cauchy_per_channel(p, s, Lambda)                     # (D, L)
    cc = cauchy_per_channel(C.conj(), s, Lambda)               # (D, L)

    # Woodbury correction for the rank-1 term, fully vectorised over D.
    k_f = cc * cb - (cc * cp) * (cq * cb) / (1 + cq * cp)     # (D, L)
    # Bilinear ZOH factor, per channel.
    k_f = k_f * (2 / (1 + omega)).unsqueeze(0) * dt.unsqueeze(-1)  # (D, L)

    k = torch.fft.ifft(k_f, n=L, dim=-1).real                 # (D, L)
    return k


# ---------------------------------------------------------------------------
# S4 Layer
# ---------------------------------------------------------------------------

class S4Layer(nn.Module):
    """
    Full S4 (Gu et al. 2022) with HiPPO-LegS initialisation and NPLR kernel.

    State matrix: A = diag(Λ) - p q^*  (Normal Plus Low-Rank)
    Kernel computed via Cauchy dot products + Woodbury identity in frequency
    domain, VECTORISED across all d_model channels (no Python for-loop).
    Convolution applied via FFT.
    """

    def __init__(self, d_model: int, d_state: int = 64, dt: float = 0.01):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state

        # Initialise NPLR decomposition from HiPPO once, then make learnable
        Lambda, p, q, _ = _nplr_hippo(d_state)
        # Store real/imag parts separately (nn.Parameter doesn't support complex)
        self.Lambda_re = nn.Parameter(Lambda.real.unsqueeze(0).expand(d_model, -1).clone())
        self.Lambda_im = nn.Parameter(Lambda.imag.unsqueeze(0).expand(d_model, -1).clone())
        self.p_re = nn.Parameter(p.real.unsqueeze(0).expand(d_model, -1).clone())
        self.p_im = nn.Parameter(p.imag.unsqueeze(0).expand(d_model, -1).clone())
        self.q_re = nn.Parameter(q.real.unsqueeze(0).expand(d_model, -1).clone())
        self.q_im = nn.Parameter(q.imag.unsqueeze(0).expand(d_model, -1).clone())

        self.B_re = nn.Parameter(torch.randn(d_model, d_state) * 0.01)
        self.B_im = nn.Parameter(torch.randn(d_model, d_state) * 0.01)
        self.C_re = nn.Parameter(torch.randn(d_model, d_state) * 0.01)
        self.C_im = nn.Parameter(torch.randn(d_model, d_state) * 0.01)

        self.log_dt = nn.Parameter(math.log(dt) * torch.ones(d_model))
        self.D = nn.Parameter(torch.ones(d_model))

    def _kernel(self, L: int) -> torch.Tensor:
        """Compute convolution kernel: (d_model, L) real. Fully vectorised
        over channels — no per-channel Python loop."""
        Lambda = torch.complex(self.Lambda_re, self.Lambda_im)  # (D, S)
        p      = torch.complex(self.p_re, self.p_im)
        q      = torch.complex(self.q_re, self.q_im)
        B      = torch.complex(self.B_re, self.B_im)            # (D, S)
        C      = torch.complex(self.C_re, self.C_im)            # (D, S)

        return _s4_kernel_batched(Lambda, p, q, B, C, self.log_dt, L)  # (D, L)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, L, _ = x.shape

        k = self._kernel(L)                                     # (D, L)

        # FFT convolution: x is (B, L, D), convolve each channel independently
        fft_len = 1 << (2 * L - 1).bit_length()
        x_f = torch.fft.rfft(x.permute(0, 2, 1), n=fft_len)   # (B, D, F)
        k_f = torch.fft.rfft(k, n=fft_len)                     # (D, F)
        y   = torch.fft.irfft(x_f * k_f.unsqueeze(0), n=fft_len)[..., :L]  # (B, D, L)

        return y.permute(0, 2, 1) + self.D * x                 # (B, L, D)



class S4Block(nn.Module):
    def __init__(self, d_model: int, d_state: int = 64, dt: float = 0.01):
        super().__init__()
        self.norm  = nn.LayerNorm(d_model)
        self.s4    = S4Layer(d_model, d_state, dt)
        self.ff    = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.s4(self.norm(x)) + x
        return self.ff(self.norm2(x)) + x


class S4ForecastModel(nn.Module):
    """
    S4-based forecast model with full HiPPO-LegS / NPLR kernel.
    Input:  (B, context_length, 1)
    Output: (B, prediction_length)
    """

    def __init__(self, prediction_length: int, d_model: int = 64,
                 d_state: int = 64, dt: float = 0.01, num_layers: int = 2):
        super().__init__()
        self.input_proj = nn.Linear(1, d_model)
        self.blocks = nn.ModuleList([
            S4Block(d_model, d_state, dt) for _ in range(num_layers)
        ])
        self.out_proj = nn.Linear(d_model, prediction_length)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(x)
        for block in self.blocks:
            h = block(h)
        return self.out_proj(h[:, -1, :])

    def param_num(self) -> int:
        return sum(p.numel() for p in self.parameters())