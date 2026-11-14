"""Counterfactual target audit (paper Section IV-F, Eqs. (17)-(18), Fig. 5): does the named optical branch read the drone or its
time slot? Accuracy within the session cannot tell the two apart, so two interventions move the drone without its slot.

  background-only run   the central 32 x 32 of every crop in both sessions is replaced by the border median; the whole method runs
                        again. Retained gain G_bg = (Acc_bg - Acc_bg^RF) / (Acc - Acc^RF), Eq. (17): near 0 for a branch that reads
                        the drone, near 1 for one that reads the slot.
  target transplant     after naming, each deployment crop loses its drone and receives the drone of window k of the record
                        (pi(y), same session, round and action) on its own background. F_tgt, Eq. (18): the share of records whose
                        optical decision is pi(y). tp1 / tp2: pi(y) = the identity 1 / 2 places later in IDS; tp0, the operation
                        control, pastes the same drone back from window k + K/2 of the same record.
"""
from contextlib import contextmanager

import numpy as np
import torch
import torch.nn.functional as F

IDS = ["T0001", "T0010", "T0011", "T0100", "T0101"]


# ------------------------------------------------------------------ counterfactual crops (no fitting)
def donors(w, shift):
    """Donor window of every window: same session D, action A, round S and window k, identity shifted by `shift` in IDS.
    shift 0: the same record, window k + K/2 (the same drone half a record later)."""
    key = {(r.T, r.D, r.A, r.S, r.k): i for i, r in enumerate(w.itertuples())}; K = int(w.k.max()) + 1
    if shift == 0: return np.array([key[(r.T, r.D, r.A, r.S, (r.k + K // 2) % K)] for r in w.itertuples()])
    return np.array([key[(IDS[(IDS.index(r.T) + shift) % 5], r.D, r.A, r.S, r.k)] for r in w.itertuples()])


def nmed(x):
    """Median along the last axis, mean of the two middle values for an even count (as numpy)."""
    s = x.sort(-1).values; n = s.shape[-1]
    return s[..., n // 2] if n % 2 else 0.5 * (s[..., n // 2 - 1] + s[..., n // 2])


def border_median(t):
    """Per-channel median of the top and bottom 8 rows of crops t (n, C, 64, 64)."""
    return nmed(torch.cat([t[:, :, :8].flatten(2), t[:, :, -8:].flatten(2)], 2))


def target_mask(g):
    """The label-free target mask of tokens_optical.target_mask on grey crops g (n, 64, 64), in torch. Returns (mask, area)."""
    b = torch.cat([g[:, :8].flatten(1), g[:, -8:].flatten(1)], 1)
    med = nmed(b); mad = nmed((b - med[:, None]).abs())
    m = (g - med[:, None, None]).abs() > (5 * 1.4826 * mad).clamp(min=0.04)[:, None, None]
    m[:, :16] = False; m[:, -16:] = False; m[:, :, :16] = False; m[:, :, -16:] = False
    return m, m.sum((1, 2))


def background_only(t):
    """The central 32 x 32 (the target and its surroundings) replaced by the border median."""
    y = t.clone(); y[:, :, 16:48, 16:48] = border_median(t)[:, :, None, None]
    return y


def transplant(tr, td):
    """Recipient crops tr, donor crops td (n, C, 64, 64) in [0, 1]: the recipient's target mask, dilated by 2 px, is filled with
    the recipient's border median; the donor's dilated target is pasted as its contrast against the donor's border median."""
    mr = F.max_pool2d(target_mask(tr.mean(1))[0][:, None].float(), 5, 1, 2) > 0
    md = F.max_pool2d(target_mask(td.mean(1))[0][:, None].float(), 5, 1, 2) > 0
    br, bd = border_median(tr)[:, :, None, None], border_median(td)[:, :, None, None]
    y = torch.where(mr, br.expand_as(tr), tr)
    return torch.where(md, (br + td - bd).clamp(0, 1), y)


def scale_norm(x, area, out=16):
    side = (4 * area.clamp(min=1).float().sqrt()).clamp(8, 64).long(); res = torch.zeros(len(x), out, out, device=x.device)
    for s in side.unique().tolist():
        i = (side == s).nonzero()[:, 0]; a = 32 - s // 2
        res[i] = F.interpolate(x[i, a:a + s, a:a + s][:, None], size=(out, out), mode="area")[:, 0]
    return res


def appearance_token(t):
    """The 512-d appearance token of tokens_optical.appearance_token computed on (counterfactual) crops t (n, C, 64, 64)."""
    g = t.mean(1); m, area = target_mask(g)
    zn = lambda f: (f.flatten(1) - f.flatten(1).mean(1, keepdim=True)) / (f.flatten(1).std(1, keepdim=True) + 1e-3)
    return torch.cat([zn(scale_norm(g, area)), scale_norm(m.float(), area).flatten(1)], 1)


# ------------------------------------------------------------------ evaluation after naming and the two metrics
@contextmanager
def swap_tokens(X, te, path):
    """Replaces, on the deployment windows te only, every token (and its availability) that the file `path` holds; restores them
    on exit. The standardisation keeps the enrolment statistics."""
    T = dict(np.load(path)); keep = {}
    for k, val in T.items():
        if k not in X.aux: continue
        keep[k] = X.aux[k].clone(); X.aux[k][te] = torch.from_numpy(val).to(X.aux[k].device)[te].to(X.aux[k].dtype)
        if k.endswith("_ok") and k[:-3] in X.tokens:
            ti = X.tokens.index(k[:-3]); keep[("ok", ti)] = X.ok[:, ti].clone(); X.ok[te, ti] = X.aux[k][te].bool()
    try:
        yield
    finally:
        for k, val in keep.items():
            if isinstance(k, tuple): X.ok[:, k[1]] = val
            else: X.aux[k] = val


def follow_target(pred_opt, true, shift):
    """Eq. (18): share of records whose optical decision is the donor identity pi(y) = (y + shift) mod N (record-level arrays)."""
    return float((pred_opt == (true + shift) % 5).float().mean())


def retained_gain(acc, acc_rf, acc_bg, acc_bg_rf):
    """Eq. (17): fused accuracy and RF-anchor accuracy of the clean run and of the background-only run."""
    return (acc_bg - acc_bg_rf) / (acc - acc_rf)
