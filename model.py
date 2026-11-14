"""Token fusion network (paper Section IV-C, Eqs. (8)-(9), Fig. 3).

One perceptron per token source, a learned source-type embedding, availability gating, a class token and two pre-norm Transformer
layers. The RF view, the optical view and the full view are readouts of the same network, obtained by masking tokens.
"""
import torch
import torch.nn as nn

D = 256                                                  # embedding width


def mlp(din):
    """phi_tau of Eq. (8): din -> 128 -> GELU -> dropout 0.1 -> 256."""
    return nn.Sequential(nn.Linear(din, 128), nn.GELU(), nn.Dropout(0.1), nn.Linear(128, D))


class AttnLayer(nn.Module):
    """Pre-norm Transformer layer: 4 heads, width 256, feed-forward 512, dropout 0.1. Attention is written as explicit matrix
    products because the fused kernels of nn.TransformerEncoder made training diverge on the MUSA accelerator."""

    def __init__(self, d=D, h=4, ff=512, p=0.1):
        super().__init__()
        self.h, self.qkv, self.o = h, nn.Linear(d, 3 * d), nn.Linear(d, d)
        self.n1, self.n2, self.drop = nn.LayerNorm(d), nn.LayerNorm(d), nn.Dropout(p)
        self.ff = nn.Sequential(nn.Linear(d, ff), nn.GELU(), nn.Dropout(p), nn.Linear(ff, d))

    def forward(self, x, pad):
        B, L, d = x.shape
        q, k, v = self.qkv(self.n1(x)).view(B, L, 3, self.h, d // self.h).permute(2, 0, 3, 1, 4)
        att = (q @ k.transpose(-1, -2)) / (d // self.h) ** 0.5
        att = self.drop(att.masked_fill(pad[:, None, None, :], -1e4).softmax(-1))      # unavailable tokens are never attended
        x = x + self.drop(self.o((att @ v).transpose(1, 2).reshape(B, L, d)))
        return x + self.drop(self.ff(self.n2(x)))


class GradReverse(torch.autograd.Function):
    """GRL_lambda of Eq. (12): identity forward, gradient multiplied by -lambda backward."""

    @staticmethod
    def forward(ctx, x, lam):
        ctx.lam = lam
        return x.view_as(x)

    @staticmethod
    def backward(ctx, g):
        return -ctx.lam * g, None


class TokenFusion(nn.Module):
    def __init__(self, tokens, dims, n=5):
        super().__init__()
        self.tokens = tokens
        self.enc = nn.ModuleDict({t: mlp(dims[t]) for t in tokens})                      # phi_tau, one per source
        self.temb = nn.Parameter(torch.randn(len(tokens), D) * 0.02)                    # chi_tau, source-type embedding
        self.cls = nn.Parameter(torch.zeros(1, 1, D))                                   # class token e^0
        self.tf = nn.ModuleList([AttnLayer(), AttnLayer()])
        self.out = nn.Sequential(nn.LayerNorm(D), nn.Dropout(0.3), nn.Linear(D, n))      # W LN(o^0), Eq. (9)
        self.dom = nn.Sequential(nn.Linear(D, 128), nn.ReLU(), nn.Linear(128, 1))       # session discriminator d_omega, Eq. (12)

    def forward(self, x, emb=False):
        """x = (list of standardised tokens, availability a (B, T) bool). Returns the identity logits of every window; with emb=True
        also LN(o^0), the input of the session discriminator."""
        xs, av = x; B = len(av)
        z = torch.stack([self.enc[t](xi) for t, xi in zip(self.tokens, xs)], 1)
        z = (z + self.temb[None]) * av[..., None].float()                               # Eq. (8)
        pad = torch.cat([torch.zeros(B, 1, dtype=torch.bool, device=z.device), ~av], 1)  # the class token is always present
        h = torch.cat([self.cls.expand(B, -1, -1), z], 1)
        for layer in self.tf: h = layer(h, pad)
        return (self.out(h[:, 0]), self.out[0](h[:, 0])) if emb else self.out(h[:, 0])
