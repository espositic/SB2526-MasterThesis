"""
Blocco Hydra (SSM bidirezionale "quasi-separabile") in PyTorch puro.

Port di modules/hydra/modules/hydra.py di XLSR-MamBo
(https://github.com/saki-ciallo/MamBo-for-ADD, licenza MIT), a sua volta basato
su Hydra (Hwang et al., 2024, https://github.com/goombalab/hydra) e su Mamba2.

L'originale usa i kernel Triton di mamba-ssm (solo Linux). Qui lo scan è
calcolato nella forma "SSD" quadratica: con clip di ~4 s le sequenze XLS-R
sono di ~200 frame, quindi la matrice L×L è piccola e il costo è trascurabile.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    """RMSNorm come in mamba-ssm (peso inizializzato a 1, nessun bias)."""

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x * self.weight.float()).to(dtype)


class GatedRMSNorm(RMSNorm):
    """RMSNorm seguita dal gate SiLU(z) (norm_before_gate=True in mamba-ssm)."""

    def forward(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        return (super().forward(x).float() * F.silu(z.float())).to(dtype)


def ssd_scan(x, dt, A, B, C):
    """
    Scan SSM causale (equivalente a mamba_chunk_scan_combined con D=None, z=None).

    h_t = exp(dt_t·A) h_{t-1} + dt_t · B_t x_t      y_t = C_t · h_t

    x: (b, l, h, p)   dt: (b, l, h) già passato da softplus   A: (h,)
    B, C: (b, l, n)   (un solo gruppo)
    Ritorna y: (b, l, h, p)
    """
    l = x.shape[1]
    a = (dt * A).float()                                # (b, l, h)
    cs = torch.cumsum(a, dim=1).transpose(1, 2)         # (b, h, l)
    # decadimento tra i e j (i >= j): exp(somma di a da j+1 a i)
    seg = cs.unsqueeze(-1) - cs.unsqueeze(-2)           # (b, h, l, l)
    causal = torch.ones(l, l, dtype=torch.bool, device=x.device).tril()
    decay = torch.exp(seg.masked_fill(~causal, float("-inf")))
    cb = torch.einsum("bin,bjn->bij", C.float(), B.float())  # (b, l, l)
    m = decay * cb.unsqueeze(1)                          # (b, h, l, l)
    xdt = x.float() * dt.float().unsqueeze(-1)           # (b, l, h, p)
    y = torch.einsum("bhij,bjhp->bihp", m, xdt)
    return y.to(x.dtype)


class Hydra(nn.Module):
    """
    Stessi parametri e stessa inizializzazione dell'originale (ngroups=1,
    init_states non appresi, nessun dt_limit): i pesi sono intercambiabili.
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 64,
        d_conv: int = 7,
        expand: int = 2,
        headdim: int = 64,
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        dt_init_floor: float = 1e-4,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = expand * d_model
        self.headdim = headdim
        assert self.d_inner % headdim == 0, "d_inner deve essere multiplo di headdim"
        self.nheads = self.d_inner // headdim

        # Ordine delle uscite di in_proj: [z, x, B, C, dt] (B e C doppi: avanti e indietro)
        d_in_proj = 2 * self.d_inner + 2 * (2 * d_state) + 2 * self.nheads
        self.in_proj = nn.Linear(d_model, d_in_proj, bias=False)

        conv_dim = self.d_inner + 2 * (2 * d_state)
        self.conv1d = nn.Conv1d(conv_dim, conv_dim, kernel_size=d_conv, groups=conv_dim,
                                padding=d_conv // 2, bias=True)

        # Bias di dt: inversa della softplus di valori log-uniformi in [dt_min, dt_max]
        dt = torch.exp(torch.rand(self.nheads) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min))
        dt = torch.clamp(dt, min=dt_init_floor)
        self.dt_bias = nn.Parameter(dt + torch.log(-torch.expm1(-dt)))

        self.A_log = nn.Parameter(torch.zeros(self.nheads))  # A = -exp(0) = -1
        self.D = nn.Parameter(torch.ones(self.nheads))
        self.fc_D = nn.Linear(self.d_inner, self.nheads, bias=False)

        self.norm = GatedRMSNorm(self.d_inner, eps=1e-5)
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        """u: (b, l, d_model) → (b, l, d_model)"""
        batch = u.shape[0]
        n2 = 2 * self.d_state
        A = -torch.exp(self.A_log.float())

        z, xBC, dt = torch.split(self.in_proj(u), [self.d_inner, self.d_inner + 2 * n2, 2 * self.nheads], dim=-1)

        # Le due direzioni diventano due "batch": avanti e sequenza ribaltata
        dt = torch.cat((dt[..., :self.nheads], torch.flip(dt[..., self.nheads:], (1,))), dim=0)
        dt = F.softplus(dt.float() + self.dt_bias.float())   # (2b, l, h)

        xBC = F.silu(self.conv1d(xBC.transpose(1, 2)).transpose(1, 2))
        x, BC = torch.split(xBC, [self.d_inner, 2 * n2], dim=-1)
        x_og = x
        x = torch.cat((x, torch.flip(x, (1,))), dim=0)
        BC = torch.cat((BC[..., :n2], torch.flip(BC[..., n2:], (1,))), dim=0)
        B, C = torch.split(BC, [self.d_state, self.d_state], dim=-1)

        y = ssd_scan(x.unflatten(-1, (self.nheads, self.headdim)), dt, A, B, C).flatten(-2)

        # Spostamento di un passo: ogni direzione esclude il frame corrente,
        # che viene aggiunto una sola volta dal termine "D" (quasi-separabile)
        y = torch.roll(y, shifts=1, dims=1)
        y[:, 0, :] = 0.0
        y = y[:batch] + torch.flip(y[batch:], (1,))
        d = F.linear(x_og, self.fc_D.weight, bias=self.D)      # (b, l, h)
        y = y + x_og * d.repeat_interleave(self.headdim, dim=-1)

        return self.out_proj(self.norm(y, z))
