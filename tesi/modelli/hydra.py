# Blocco Hydra (Mamba bidirezionale "quasi-separabile") in PyTorch puro, usato da XLSR-MamBo.
#
# Port di modules/hydra/modules/hydra.py di MamBo-for-ADD (github.com/saki-ciallo/MamBo-for-ADD,
# licenza MIT), a sua volta basato su Hydra (Hwang et al., 2024) e Mamba2.
# L'originale usa i kernel Triton di mamba-ssm, che su Windows non ci sono: qui lo scan è
# calcolato con una matrice L x L. Con clip di ~4 s le sequenze sono di ~200 frame, quindi va bene.

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        tipo = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x * self.weight.float()).to(tipo)


class GatedRMSNorm(RMSNorm):
    """RMSNorm seguita dal gate SiLU(z)."""

    def forward(self, x, z):
        tipo = x.dtype
        return (super().forward(x).float() * F.silu(z.float())).to(tipo)


def ssd_scan(x, dt, A, B, C):
    """
    Scan dello state space model:  h_t = exp(dt_t A) h_(t-1) + dt_t B_t x_t,   y_t = C_t h_t
    x: (b, l, teste, p)   dt: (b, l, teste)   A: (teste,)   B, C: (b, l, n)
    """
    l = x.shape[1]
    a = (dt * A).float()
    cs = torch.cumsum(a, dim=1).transpose(1, 2)            # (b, teste, l)
    seg = cs.unsqueeze(-1) - cs.unsqueeze(-2)              # decadimento tra il passo j e il passo i
    causale = torch.ones(l, l, dtype=torch.bool, device=x.device).tril()
    decadimento = torch.exp(seg.masked_fill(~causale, float("-inf")))
    cb = torch.einsum("bin,bjn->bij", C.float(), B.float())
    m = decadimento * cb.unsqueeze(1)
    xdt = x.float() * dt.float().unsqueeze(-1)
    y = torch.einsum("bhij,bjhp->bihp", m, xdt)
    return y.to(x.dtype)


class Hydra(nn.Module):
    """Stessi parametri e stessa inizializzazione dell'originale."""

    def __init__(self, d_model, d_state=64, d_conv=7, expand=2, headdim=64, dt_min=0.001, dt_max=0.1):
        super().__init__()
        self.d_state = d_state
        self.d_inner = expand * d_model
        self.headdim = headdim
        self.nheads = self.d_inner // headdim

        # uscite di in_proj: [z, x, B, C, dt] (B, C e dt doppi: una copia per direzione)
        self.in_proj = nn.Linear(d_model, 2 * self.d_inner + 4 * d_state + 2 * self.nheads, bias=False)
        conv_dim = self.d_inner + 4 * d_state
        self.conv1d = nn.Conv1d(conv_dim, conv_dim, kernel_size=d_conv, groups=conv_dim, padding=d_conv // 2)

        # bias di dt: inversa della softplus di valori log-uniformi tra dt_min e dt_max
        dt = torch.exp(torch.rand(self.nheads) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min))
        dt = torch.clamp(dt, min=1e-4)
        self.dt_bias = nn.Parameter(dt + torch.log(-torch.expm1(-dt)))
        self.A_log = nn.Parameter(torch.zeros(self.nheads))
        self.D = nn.Parameter(torch.ones(self.nheads))
        self.fc_D = nn.Linear(self.d_inner, self.nheads, bias=False)
        self.norm = GatedRMSNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def forward(self, u):
        b = u.shape[0]
        n2 = 2 * self.d_state
        A = -torch.exp(self.A_log.float())
        z, xBC, dt = torch.split(self.in_proj(u), [self.d_inner, self.d_inner + 2 * n2, 2 * self.nheads], dim=-1)

        # le due direzioni diventano due metà del batch: sequenza normale e sequenza ribaltata
        dt = torch.cat((dt[..., :self.nheads], torch.flip(dt[..., self.nheads:], (1,))), dim=0)
        dt = F.softplus(dt.float() + self.dt_bias.float())

        xBC = F.silu(self.conv1d(xBC.transpose(1, 2)).transpose(1, 2))
        x, BC = torch.split(xBC, [self.d_inner, 2 * n2], dim=-1)
        x_originale = x
        x = torch.cat((x, torch.flip(x, (1,))), dim=0)
        BC = torch.cat((BC[..., :n2], torch.flip(BC[..., n2:], (1,))), dim=0)
        B, C = torch.split(BC, [self.d_state, self.d_state], dim=-1)

        y = ssd_scan(x.unflatten(-1, (self.nheads, self.headdim)), dt, A, B, C).flatten(-2)

        # ogni direzione esclude il frame corrente, che viene aggiunto una volta sola dal termine D
        y = torch.roll(y, shifts=1, dims=1)
        y[:, 0, :] = 0.0
        y = y[:b] + torch.flip(y[b:], (1,))
        d = F.linear(x_originale, self.fc_D.weight, bias=self.D)
        y = y + x_originale * d.repeat_interleave(self.headdim, dim=-1)
        return self.out_proj(self.norm(y, z))
