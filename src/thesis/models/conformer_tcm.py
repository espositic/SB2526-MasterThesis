"""
XLSR-Conformer con Temporal-Channel Modeling (TCM) nell'attenzione:
Truong, Tao, Nguyen, Luong, Lee, Chng, "Temporal-Channel Modeling in Multi-head
Self-Attention for Synthetic Speech Detection", Interspeech 2024, arXiv 2406.17376.
Codice originale: https://github.com/ductuantruong/tcm_add (licenza MIT, vedi
THIRD_PARTY_NOTICES.md); a sua volta basato su XLSR-Conformer di Rosello et al.

Port del modello (model.py e conformer.py dell'originale):
XLS-R 300M → lineare a 144 + BatchNorm + SELU → 4 blocchi Conformer con token CLS
e "head token" TCM nell'attenzione → lineare a 2 classi sul token CLS.

Variante kan=True, "XLSR-Kanformer" (Phuong e Tran, IEEE AVSS 2025,
arXiv 2510.06706, codice non pubblicato, re-implementata dal paper):
layer ChebyKAN (polinomi di Chebyshev) al posto della proiezione delle feature
SSL, dei due lineari di ogni feed-forward e delle convoluzioni puntuali del
modulo convolutivo. Il paper non dà il grado dei polinomi né dice se cambia
anche la convoluzione depthwise: qui grado 4 (kan_degree) e depthwise invariata.
Prima del ChebyKAN di proiezione c'è una LayerNorm, perché ChebyKAN comprime
l'ingresso con tanh e le feature di XLS-R hanno scala ampia (scelta nostra).

Differenze rispetto all'originale, comuni agli altri modelli del progetto:
XLS-R da Hugging Face invece che da fairseq (stessi pesi), CNN iniziale
congelata, gradient checkpointing, niente einops; le convoluzioni 1×1 sono
scritte come lineari (stessa operazione).

Convenzione del progetto: uscita [logit bonafide, logit spoof].
"""

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from thesis.models.ssl_frontend import XLSR_300M, load_ssl, normalize_wave


@dataclass
class TCMConfig:
    ssl_name: str = XLSR_300M
    emb_size: int = 144
    heads: int = 4
    kernel_size: int = 31
    n_encoders: int = 4
    ff_mult: int = 4
    conv_expansion: int = 2
    kan: bool = False              # True = variante XLSR-Kanformer
    kan_degree: int = 4
    freeze_ssl: bool = False
    gradient_checkpointing: bool = True


class ChebyKAN(nn.Module):
    """
    Layer KAN con polinomi di Chebyshev: y_o = Σ_i Σ_d c[i,o,d] · T_d(tanh(x_i)),
    ricorrenza T0 = 1, T1 = x, T_m = 2x·T_{m−1} − T_{m−2} (eq. 3 di XLSR-Kanformer).
    Inizializzazione come in ChebyKAN (SynodicMonth/ChebyKAN): N(0, 1/(d_in·(grado+1))).
    """

    def __init__(self, d_in: int, d_out: int, degree: int = 4):
        super().__init__()
        self.degree = degree
        self.coeffs = nn.Parameter(torch.empty(d_in, d_out, degree + 1))
        nn.init.normal_(self.coeffs, mean=0.0, std=1 / (d_in * (degree + 1)))

    def forward(self, x):
        x = torch.tanh(x)
        terms = [torch.ones_like(x), x]
        for _ in range(2, self.degree + 1):
            terms.append(2 * x * terms[-1] - terms[-2])
        basis = torch.stack(terms[:self.degree + 1], dim=-1)            # (…, d_in, grado+1)
        return torch.einsum("...id,iod->...o", basis, self.coeffs.to(basis.dtype))


def sinusoidal_embedding(n_positions: int, dim: int) -> torch.Tensor:
    pos = torch.arange(n_positions, dtype=torch.float32)[:, None]
    i = torch.arange(dim, dtype=torch.float32)[None, :]
    pe = pos / 10000 ** (2 * (i // 2) / dim)
    pe[:, 0::2] = torch.sin(pe[:, 0::2])
    pe[:, 1::2] = torch.cos(pe[:, 1::2])
    return pe.unsqueeze(0)


class TCMAttention(nn.Module):
    """
    MHSA con "head token" (TCM): per ogni testa un token riassume i canali di quella
    testa mediati sul tempo; i token entrano nell'attenzione insieme ai frame e alla
    fine vengono fusi nel token CLS, insieme alla media dei frame.
    """

    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.heads = heads
        dim_head = dim // heads
        self.scale = dim_head ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)
        self.ht_proj = nn.Linear(dim_head, dim)
        self.ht_norm = nn.LayerNorm(dim_head)
        self.pos_embed = nn.Parameter(torch.zeros(1, heads, dim))

    def forward(self, x):
        b, n, c = x.shape
        h = self.heads
        # head token: media sul tempo dei canali di ogni testa → (b, h, dim)
        ht = x.reshape(b, n, h, c // h).permute(0, 2, 1, 3).mean(dim=2)
        ht = self.ht_proj(ht).reshape(b, -1, h, c // h)
        ht = F.gelu(self.ht_norm(ht)).flatten(2) + self.pos_embed
        x = torch.cat([x, ht], dim=1)

        q, k, v = self.qkv(x).reshape(b, n + h, 3, h, c // h).permute(2, 0, 3, 1, 4)
        attn = ((q @ k.transpose(-2, -1)) * self.scale).softmax(dim=-1)
        x = self.proj((attn @ v).transpose(1, 2).reshape(b, n + h, c))

        cls, frames, ht = torch.split(x, [1, n - 1, h], dim=1)
        cls = cls + ht.mean(dim=1, keepdim=True) + frames.mean(dim=1, keepdim=True)
        return torch.cat([cls, frames], dim=1)


class FeedForward(nn.Module):
    def __init__(self, dim: int, mult: int, kan: bool, degree: int):
        super().__init__()
        if kan:
            self.net = nn.Sequential(ChebyKAN(dim, dim * mult, degree), ChebyKAN(dim * mult, dim, degree))
        else:
            self.net = nn.Sequential(nn.Linear(dim, dim * mult), nn.SiLU(), nn.Linear(dim * mult, dim))

    def forward(self, x):
        return self.net(x)


class ConvModule(nn.Module):
    """LayerNorm → puntuale + GLU → depthwise → BatchNorm → Swish → puntuale."""

    def __init__(self, dim: int, expansion: int, kernel_size: int, kan: bool, degree: int):
        super().__init__()
        inner = dim * expansion
        self.norm = nn.LayerNorm(dim)
        self.pointwise_in = ChebyKAN(dim, inner * 2, degree) if kan else nn.Linear(dim, inner * 2)
        pad = kernel_size // 2
        self.padding = (pad, pad - (kernel_size + 1) % 2)
        self.depthwise = nn.Conv1d(inner, inner, kernel_size, groups=inner)
        self.bn = nn.BatchNorm1d(inner)
        self.pointwise_out = ChebyKAN(inner, dim, degree) if kan else nn.Linear(inner, dim)

    def forward(self, x):                                    # x: (b, n, dim)
        x = F.glu(self.pointwise_in(self.norm(x)), dim=-1)    # (b, n, inner)
        x = self.depthwise(F.pad(x.transpose(1, 2), self.padding))
        x = F.silu(self.bn(x)).transpose(1, 2)
        return self.pointwise_out(x)


class ConformerBlock(nn.Module):
    def __init__(self, dim: int, heads: int, ff_mult: int, expansion: int, kernel_size: int,
                 kan: bool, degree: int):
        super().__init__()
        self.ff1_norm, self.ff1 = nn.LayerNorm(dim), FeedForward(dim, ff_mult, kan, degree)
        self.attn_norm, self.attn = nn.LayerNorm(dim), TCMAttention(dim, heads)
        self.conv = ConvModule(dim, expansion, kernel_size, kan, degree)
        self.ff2_norm, self.ff2 = nn.LayerNorm(dim), FeedForward(dim, ff_mult, kan, degree)
        self.post_norm = nn.LayerNorm(dim)

    def forward(self, x):
        x = x + 0.5 * self.ff1(self.ff1_norm(x))
        x = x + self.attn(self.attn_norm(x))
        x = x + self.conv(x)
        x = x + 0.5 * self.ff2(self.ff2_norm(x))
        return self.post_norm(x)


class XLSRConformerTCM(nn.Module):
    def __init__(self, cfg: TCMConfig):
        super().__init__()
        self.cfg = cfg
        self.ssl, self.normalize = load_ssl(cfg.ssl_name, freeze=cfg.freeze_ssl,
                                            gradient_checkpointing=cfg.gradient_checkpointing and not cfg.freeze_ssl)
        d_ssl, d = self.ssl.config.hidden_size, cfg.emb_size
        if cfg.kan:
            self.proj = nn.Sequential(nn.LayerNorm(d_ssl), ChebyKAN(d_ssl, d, cfg.kan_degree))
        else:
            self.proj = nn.Linear(d_ssl, d)
            self.first_bn = nn.BatchNorm2d(1)
        self.register_buffer("pos_emb", sinusoidal_embedding(10000, d), persistent=False)
        self.cls_token = nn.Parameter(torch.rand(1, 1, d))
        self.blocks = nn.ModuleList([
            ConformerBlock(d, cfg.heads, cfg.ff_mult, cfg.conv_expansion, cfg.kernel_size, cfg.kan, cfg.kan_degree)
            for _ in range(cfg.n_encoders)])
        self.classifier = nn.Linear(d, 2)

    def train(self, mode: bool = True):
        super().train(mode)
        if self.cfg.freeze_ssl:
            self.ssl.eval()
        return self

    def forward(self, wave: torch.Tensor) -> torch.Tensor:
        """wave: (b, campioni) a 16 kHz → logit (b, 2) = [bonafide, spoof]"""
        if self.normalize:
            wave = normalize_wave(wave)
        with torch.set_grad_enabled(torch.is_grad_enabled() and not self.cfg.freeze_ssl):
            x = self.ssl(wave).last_hidden_state                  # (b, frame, 1024)
        if self.cfg.kan:
            x = self.proj(x)
        else:
            x = F.selu(self.first_bn(self.proj(x).unsqueeze(1)).squeeze(1))
        x = x + self.pos_emb[:, :x.shape[1]].to(x.dtype)
        x = torch.cat([self.cls_token.expand(len(x), -1, -1).to(x.dtype), x], dim=1)
        for block in self.blocks:
            x = block(x)
        return self.classifier(x[:, 0])
