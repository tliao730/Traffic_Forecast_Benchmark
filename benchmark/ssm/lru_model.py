import math
import torch
import torch.nn as nn

# The discrete Lyapunov solves in LRU.compute_gramians() invert (I4 - A(x)A),
# whose eigenvalues are {|lam|^2, |lam|^2, |lam|^2 e^{+-2i.theta}} -- the matrix is
# exactly singular once |lam| reaches 1. Training can drive nu_log low enough
# that lam_abs = exp(-exp(nu_log)) rounds to exactly 1.0 in float32
# (eps ~ 1.19e-7); that killed the CA/2018 run at its epoch-9 compression.
# Cap |lam| in the compression path so 1 - |lam|^2 >= LYAP_MARGIN.
LYAP_MARGIN = 1e-4


def _ssm_fft_conv_complex(x: torch.Tensor, lam: torch.Tensor,
                           B_re: torch.Tensor, B_im: torch.Tensor,
                           C: torch.Tensor) -> torch.Tensor:
    """
    LTI SSM via FFT convolution (complex-valued state, real output).
        h_t = lam * h_{t-1} + (B_re + i*B_im) @ x_t
        y_t = C @ [Re(h_t); Im(h_t)]

    x:    (B, L, D)
    lam:  (D, S) complex
    B_re: (S, D), B_im: (S, D)  — weight matrices
    C:    (D, 2S)               — weight matrix
    Returns y: (B, L, D)
    """
    _, L, _ = x.shape

    t = torch.arange(L, device=lam.device, dtype=lam.real.dtype)
    lam_t = lam.unsqueeze(-1) ** t.unsqueeze(0).unsqueeze(0)  # (D, S, L) complex

    Bx = torch.complex(x @ B_re.T, x @ B_im.T)  # (B, L, S) complex

    fft_len = 1 << (2 * L - 1).bit_length()
    Bx_f  = torch.fft.fft(Bx.permute(0, 2, 1), n=fft_len)   # (B, S, F)
    lam_f = torch.fft.fft(lam_t, n=fft_len)                  # (D, S, F)

    conv = lam_f.unsqueeze(0) * Bx_f.unsqueeze(1)            # (B, D, S, F)
    h_s  = torch.fft.ifft(conv, n=fft_len)[..., :L]          # (B, D, S, L) complex

    # flatten Re/Im and project with C
    h_cat = torch.cat([h_s.real, h_s.imag], dim=2)           # (B, D, 2S, L)
    # C: (D, 2S) -> sum over 2S dim
    y = (C.unsqueeze(0).unsqueeze(-1) * h_cat).sum(dim=2)    # (B, D, L)
    return y.permute(0, 2, 1)                                 # (B, L, D)


class LRULayer(nn.Module):
    """
    Linear Recurrent Unit (Orvieto et al. 2023), solved via FFT convolution.

    Key design choices:
      - Complex-valued diagonal state: richer frequency representation
      - Ring initialisation: |lambda| ~ Uniform(lam_min, lam_max) and
        theta ~ Uniform(theta_min, theta_max), so each channel starts with
        both a distinct memory timescale AND a distinct oscillation frequency
      - Gamma-normalised B so that steady-state ||h_t|| stays bounded
        regardless of |lambda| (Orvieto et al. 2023, Sec. 3.2)
      - B/C initialised with variance scaled by 1/dim, following the paper's
        recommendation, rather than relying on nn.Linear's default Kaiming init
      - No output gate: follows the original LRU paper exactly
      - C is treated as a fully free real-valued linear map from
        [Re(h); Im(h)] to y (not constrained to correspond to a single
        complex matrix C_complex = C_re + i*C_im). This gives the model
        more learning capacity, at the cost of compression having to act
        on the real/imag halves of C separately rather than on one
        complex C — see `compress()` below.

    -------------------------------------------------------------------
    CompreSSM: in-training compression via balanced truncation
    -------------------------------------------------------------------
    Because C is NOT constrained to a single complex matrix, we cannot
    reduce this to the standard complex-LTI Gramian recipe. Instead we
    build an equivalent REAL LTI system of size 2S (stacking Re(h) and
    Im(h) as independent real state coordinates) and apply the standard
    real discrete Lyapunov / balanced truncation recipe to that
    2S-dimensional system. This exactly respects the actual computational
    graph and requires no assumption about a hidden complex structure in C.
    """

    def __init__(self, d_model: int, d_state: int = 16,
                 lam_min: float = 0.1, lam_max: float = 0.999,
                 theta_min: float = 0.0, theta_max: float = math.pi):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state

        # --- Ring initialisation (Orvieto et al. 2023) ---
        # |lambda| ~ Uniform(lam_min, lam_max): spreads channels across
        # memory timescales so the model doesn't have to learn this from
        # a single shared starting point.
        lam_init = lam_min + (lam_max - lam_min) * torch.rand(d_model, d_state)
        self.nu_log = nn.Parameter(torch.log(-torch.log(lam_init)))

        # theta ~ Uniform(theta_min, theta_max): spreads channels across
        # oscillation frequencies. Exposed as constructor args (rather than
        # hardcoded to [0, pi]) so the range can be swept in ablations.
        theta_init = theta_min + (theta_max - theta_min) * torch.rand(d_model, d_state)
        self.theta_log = nn.Parameter(torch.log(theta_init.clamp(min=1e-6)))

        self.B_re = nn.Linear(d_model, d_state, bias=False)
        self.B_im = nn.Linear(d_model, d_state, bias=False)
        self.C    = nn.Linear(2 * d_state, d_model, bias=False)
        self.D    = nn.Parameter(torch.ones(d_model))

        # --- B/C initialisation (Orvieto et al. 2023) ---
        # Default nn.Linear init (Kaiming uniform) is tuned for generic
        # feedforward layers, not for the variance-preservation requirements
        # of a recurrent SSM. Re-initialise so output variance is governed
        # by 1/dim rather than left to PyTorch's default.
        nn.init.normal_(self.B_re.weight, mean=0.0, std=1.0 / math.sqrt(2 * d_model))
        nn.init.normal_(self.B_im.weight, mean=0.0, std=1.0 / math.sqrt(2 * d_model))
        nn.init.normal_(self.C.weight,    mean=0.0, std=1.0 / math.sqrt(d_state))

    def _discretised_params(self):
        """Shared helper: compute lambda (complex) and gamma-normalised B."""
        nu      = torch.exp(self.nu_log)
        theta   = torch.exp(self.theta_log)
        lam_abs = torch.exp(-nu)
        lam     = torch.complex(lam_abs * torch.cos(theta),
                                lam_abs * torch.sin(theta))  # (D, S)

        gamma   = torch.sqrt((1 - lam_abs ** 2).clamp(min=1e-6)).T  # (S, D)
        B_re_w  = gamma * self.B_re.weight  # (S, D)
        B_im_w  = gamma * self.B_im.weight  # (S, D)
        return lam, lam_abs, B_re_w, B_im_w

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lam, lam_abs, B_re_w, B_im_w = self._discretised_params()
        y = _ssm_fft_conv_complex(x, lam, B_re_w, B_im_w, self.C.weight)
        return y + self.D * x

    # ------------------------------------------------------------------
    # CompreSSM: in-training compression via balanced truncation
    # ------------------------------------------------------------------

    def _real_state_space_matrices(self):
        """
        Build the equivalent REAL 2-dimensional-per-state-index LTI block
        system. Each complex state index s corresponds to a 2x2 real
        rotation-scaling block:

            A_s = |lambda_s| * [[cos(theta_s), -sin(theta_s)],
                                 [sin(theta_s),  cos(theta_s)]]

        Returns:
            A_blocks: (D, S, 2, 2) real
            b_blocks: (D, S, 2)    real
        """
        lam, lam_abs, B_re_w, B_im_w = self._discretised_params()

        # Keep A strictly inside the unit circle so the Lyapunov solves in
        # compute_gramians() stay non-singular (see LYAP_MARGIN). This touches
        # the balanced-truncation analysis only -- forward() calls
        # _discretised_params() directly and is unaffected. b is deliberately
        # left as the model's real gamma-normalised B.
        lam_abs = lam_abs.clamp(max=math.sqrt(1.0 - LYAP_MARGIN))

        B_re = B_re_w.T  # (D, S)
        B_im = B_im_w.T  # (D, S)

        theta   = torch.exp(self.theta_log)
        cos, sin = torch.cos(theta), torch.sin(theta)

        A_blocks = lam_abs.unsqueeze(-1).unsqueeze(-1) * torch.stack([
            torch.stack([cos, -sin], dim=-1),
            torch.stack([sin,  cos], dim=-1),
        ], dim=-2)                                       # (D, S, 2, 2)

        b_blocks = torch.stack([B_re, B_im], dim=-1)    # (D, S, 2)
        return A_blocks, b_blocks

    @staticmethod
    def _batched_kron_2x2(A: torch.Tensor) -> torch.Tensor:
        """
        Fully vectorised Kronecker product A ⊗ A for a batch of 2x2
        matrices, with NO Python loop over the batch dimensions.

        A: (..., 2, 2) -> returns (..., 4, 4)
        """
        *batch, _, _ = A.shape
        AA = A.unsqueeze(-1).unsqueeze(-1) * A.unsqueeze(-3).unsqueeze(-3)
        AA = AA.permute(*range(len(batch)), -4, -2, -3, -1)
        return AA.reshape(*batch, 4, 4)

    def compute_gramians(self):
        """
        Compute controllability gramian P and observability gramian Q for
        each per-state 2x2 real block via the discrete Lyapunov equation:

            P: A P A^T - P + b b^T = 0  ->  vec(P) = (I4 - A⊗A)^{-1} vec(bb^T)
            Q: A^T Q A - Q + c c^T = 0  ->  vec(Q) = (I4 - A^T⊗A^T)^{-1} vec(cc^T)

        For Q, c_s is the observation vector [C_re[:,s]; C_im[:,s]] per
        output channel. We compute Q independently for each output channel
        and average the traces — this avoids sign cancellation that would
        occur if we summed C rows before computing the Gramian.

        Returns:
            P: (D, S) real — trace of per-state controllability Gramian
            Q: (D, S) real — mean over output channels of per-state
                             observability Gramian trace
        """
        A_blocks, b_blocks = self._real_state_space_matrices()   # (D,S,2,2), (D,S,2)
        D, S = A_blocks.shape[0], A_blocks.shape[1]
        device = A_blocks.device

        I4 = torch.eye(4, device=device).expand(D, S, 4, 4)

        # --- Controllability ---
        AkronA = self._batched_kron_2x2(A_blocks)                      # (D, S, 4, 4)
        bbT    = b_blocks.unsqueeze(-1) * b_blocks.unsqueeze(-2)       # (D, S, 2, 2)
        vecP   = torch.linalg.solve(I4 - AkronA,
                                    bbT.reshape(D, S, 4, 1)).squeeze(-1)
        P = vecP.reshape(D, S, 2, 2).diagonal(dim1=-2, dim2=-1).sum(-1)  # (D, S)

        # --- Observability ---
        # c_blocks: (D_out, S, 2) — one 2-vector per output channel per state.
        # Compute Q per output channel, then average traces to avoid sign
        # cancellation from summing C rows before solving the Lyapunov equation.
        C_re_half = self.C.weight[:, :self.d_state]   # (D_out, S)
        C_im_half = self.C.weight[:, self.d_state:]   # (D_out, S)
        D_out = C_re_half.shape[0]

        c_blocks = torch.stack([C_re_half, C_im_half], dim=-1)  # (D_out, S, 2)

        # Expand A^T blocks for each output channel: (D_out, D, S, 2, 2)
        AT_blocks   = A_blocks.transpose(-2, -1)                        # (D, S, 2, 2)
        ATkronAT    = self._batched_kron_2x2(AT_blocks)                 # (D, S, 4, 4)
        # Expand for D_out: (1, D, S, 4, 4) broadcast with (D_out, 1, S, 4, 4)
        ATkronAT_ex = ATkronAT.unsqueeze(0).expand(D_out, D, S, 4, 4)  # (D_out, D, S, 4, 4)
        I4_ex       = I4.unsqueeze(0).expand(D_out, D, S, 4, 4)

        # ccT: (D_out, S, 2, 2) — each output channel's outer product
        ccT = c_blocks.unsqueeze(-1) * c_blocks.unsqueeze(-2)           # (D_out, S, 2, 2)
        # Broadcast to (D_out, D, S, 4): expand ccT over the D (lambda) dimension
        ccT_ex = ccT.unsqueeze(1).expand(D_out, D, S, 2, 2)            # (D_out, D, S, 2, 2)

        vecQ = torch.linalg.solve(
            I4_ex - ATkronAT_ex,
            ccT_ex.reshape(D_out, D, S, 4, 1)
        ).squeeze(-1)                                                    # (D_out, D, S, 4)

        Q_blocks = vecQ.reshape(D_out, D, S, 2, 2)
        Q = Q_blocks.diagonal(dim1=-2, dim2=-1).sum(-1).mean(0)        # (D, S)

        return P.clamp(min=1e-8), Q.clamp(min=1e-8)

    def compute_hsv(self, P: torch.Tensor, Q: torch.Tensor) -> torch.Tensor:
        """
        Hankel singular values via trace approximation: HSV_i = sqrt(P_i * Q_i).
        This is exact when both Gramians are scalar multiples of I within each
        2x2 block (e.g. pure rotation); for general |lambda| < 1 it is a
        trace-based approximation that still correctly ranks state importance.

        Args:
            P, Q: (D, S)
        Returns:
            hsv: (D, S), sorted descending per channel
        """
        hsv = torch.sqrt((P * Q).clamp(min=0))
        hsv, _ = torch.sort(hsv, dim=-1, descending=True)
        return hsv

    @torch.no_grad()
    def compress(self, energy_threshold: float = 0.99):
        """
        Run one CompreSSM compression step: rank the d_state complex state
        dimensions by Hankel singular value, keep the smallest r that
        retains `energy_threshold` of total HSV energy, and truncate all
        parameters (nu_log, theta_log, B_re, B_im) plus BOTH halves of C
        to the new d_state = r.

        NOTE on C: because C is a free real map over [Re(h); Im(h)],
        truncating state index i means dropping column i from BOTH the
        Re-half and the Im-half of C.weight — not simply slicing the first
        r columns of the raw (2S,)-wide weight matrix.
        """
        P, Q = self.compute_gramians()
        hsv  = self.compute_hsv(P, Q)         # (D, S)

        hsv_mean = hsv.mean(dim=0)             # (S,) — average importance across channels
        total    = hsv_mean.sum()
        cumsum   = torch.cumsum(hsv_mean, dim=0)
        r = int((cumsum / (total + 1e-8) < energy_threshold).sum().item()) + 1
        r = max(1, min(r, self.d_state))

        if r >= self.d_state:
            return

        _, topk_idx = torch.topk(hsv_mean, r)
        topk_idx, _ = torch.sort(topk_idx)

        device = self.nu_log.device

        self.nu_log    = nn.Parameter(self.nu_log[:, topk_idx].detach())
        self.theta_log = nn.Parameter(self.theta_log[:, topk_idx].detach())

        old_Bre_w = self.B_re.weight.detach()   # (S, D)
        old_Bim_w = self.B_im.weight.detach()   # (S, D)
        new_Bre = nn.Linear(self.d_model, r, bias=False).to(device)
        new_Bim = nn.Linear(self.d_model, r, bias=False).to(device)
        new_Bre.weight.data = old_Bre_w[topk_idx]
        new_Bim.weight.data = old_Bim_w[topk_idx]
        self.B_re = new_Bre
        self.B_im = new_Bim

        # C.weight: (D_out, 2S) = [Re-half (D_out,S) | Im-half (D_out,S)]
        # Select topk_idx from BOTH halves, then re-concatenate to (D_out, 2r).
        old_C_w   = self.C.weight.detach()
        C_re_half = old_C_w[:, :self.d_state]
        C_im_half = old_C_w[:, self.d_state:]
        new_C_w   = torch.cat([C_re_half[:, topk_idx],
                                C_im_half[:, topk_idx]], dim=1)
        new_C = nn.Linear(2 * r, self.d_model, bias=False).to(device)
        new_C.weight.data = new_C_w
        self.C = new_C

        self.d_state = r


class LRUBlock(nn.Module):
    def __init__(self, d_model: int, d_state: int = 16,
                 lam_min: float = 0.1, lam_max: float = 0.999,
                 theta_min: float = 0.0, theta_max: float = math.pi):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.lru  = LRULayer(d_model, d_state, lam_min, lam_max, theta_min, theta_max)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lru(self.norm(x)) + x

    def compress(self, energy_threshold: float = 0.99):
        self.lru.compress(energy_threshold)


class LRUForecastModel(nn.Module):
    """
    LRU-based forecast model.
    Input:  (B, context_length, 1)
    Output: (B, prediction_length)
    """

    def __init__(self, prediction_length: int, d_model: int = 64,
                 d_state: int = 16, num_layers: int = 2,
                 lam_min: float = 0.1, lam_max: float = 0.999,
                 theta_min: float = 0.0, theta_max: float = math.pi):
        super().__init__()
        self.input_proj = nn.Linear(1, d_model)
        self.blocks = nn.ModuleList([
            LRUBlock(d_model, d_state, lam_min, lam_max, theta_min, theta_max)
            for _ in range(num_layers)
        ])
        self.out_proj = nn.Linear(d_model, prediction_length)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(x)
        for block in self.blocks:
            h = block(h)
        return self.out_proj(h[:, -1, :])

    def compress_all(self, energy_threshold: float = 0.99):
        """Run CompreSSM compression on every LRU block in the model."""
        for block in self.blocks:
            block.compress(energy_threshold)

    def param_num(self) -> int:
        return sum(p.numel() for p in self.parameters())
