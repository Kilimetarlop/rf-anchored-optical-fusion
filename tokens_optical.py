"""Optical tokens (paper Section IV-B, "Optical tokens"): per camera and window, a 512-d appearance token and a 37-d trajectory token.

Inputs: the 64 x 64 crops centred on the target track of each camera and the tracks themselves (frame, x, y, contrast[, polarity]);
detection, tracking and cropping are not part of this release. Everything is label-free and computed per crop or per track.
A camera without a crop or a track in the window gives a zero token with availability 0.
"""
import numpy as np
import torch
import torch.nn.functional as F


def grey(crops):
    """uint8 crops, (n, 64, 64, 3) visible or (n, 64, 64) infrared -> grey values in [0, 1]."""
    x = np.asarray(crops, np.float32)
    return (x.mean(-1) if x.ndim == 4 else x) / 255.0


# ------------------------------------------------------------------ appearance token (512-d)
def target_mask(x):
    """Label-free target mask: pixels of the central 32 x 32 that deviate from the median of the top and bottom 8 border rows by more
    than max(0.04, 5 x 1.4826 x MAD). Returns the mask and its area A in pixels."""
    b = np.concatenate([x[:, :8].reshape(len(x), -1), x[:, -8:].reshape(len(x), -1)], 1)
    med = np.median(b, 1); mad = np.median(np.abs(b - med[:, None]), 1)
    m = np.abs(x - med[:, None, None]) > np.maximum(0.04, 5 * 1.4826 * mad)[:, None, None]
    m[:, :16] = m[:, -16:] = False; m[:, :, :16] = m[:, :, -16:] = False
    return m, m.sum((1, 2))


def scale_norm(x, area, out=16):
    """The square of side min(max(4 sqrt(A), 8), 64) about the crop centre, area-resampled to 16 x 16: the window follows the
    target, so the target appears at a common scale and most of the scene drops out."""
    side = np.clip(4 * np.sqrt(np.maximum(area, 1)), 8, 64).astype(int); res = np.zeros((len(x), out, out), np.float32)
    for s in np.unique(side):
        i = np.where(side == s)[0]; a = 32 - s // 2
        res[i] = F.interpolate(torch.from_numpy(x[i, a:a + s, a:a + s][:, None]), size=(out, out), mode="area").numpy()[:, 0]
    return res


def zscore(x):
    """Per-crop standardisation: removes brightness and gain."""
    f = x.reshape(len(x), -1)
    return (f - f.mean(1, keepdims=True)) / (f.std(1, keepdims=True) + 1e-3)


def appearance_token(x, ok):
    """[standardised 16 x 16 target window (256), binary mask at the same scale (256)] per crop; x: grey crops (n, 64, 64),
    ok: the camera has a crop in the window."""
    m, area = target_mask(x)
    f = np.concatenate([zscore(scale_norm(x, area)), scale_norm(m.astype(np.float32), area).reshape(len(x), -1)], 1).astype(np.float32)
    f[~ok] = 0
    return f


# ------------------------------------------------------------------ trajectory token (37-d)
HALF, NB = 25, 8                                         # +-1 s at 25 frames/s; 8 spectral bands


def lg(x):
    """Log scale for scale-like quantities."""
    return np.log1p(1000 * np.abs(x))


def bands(sig, fps):
    """Normalised energy of the Hann-windowed spectrum of a detrended signal in NB equal bands up to fps / 2."""
    s = sig - sig.mean(); m = np.abs(np.fft.rfft(s * np.hanning(len(s)))); f = np.fft.rfftfreq(len(s), 1 / fps)
    edges = np.linspace(0, fps / 2, NB + 1)
    e = np.array([m[(f > lo) & (f <= hi)].sum() for lo, hi in zip(edges[:-1], edges[1:])])
    return e / (e.sum() + 1e-9)


def trajectory_token(tr, k, K, n_frames, W, H, fps, area):
    """37-d token of one camera and window k of K, or None (missing) when fewer than 10 track points lie within +-1 s of the window's
    centre frame fk = (k + 0.5) / K x n_frames. tr rows: frame, x, y, contrast z[, polarity]; positions normalised by the frame size.
      kinematics (15)        speed mean / std / max / median, signed mean vx / vy, mean |vx| / |vy|, |acceleration| mean / std,
                             mean |jerk|, heading change, log curvature, straightness, jitter after a quadratic fit
      photometry (4)         mean |z|, std of z, coefficient of variation, mean polarity
      contrast spectrum (8)  bands() of the contrast z
      vibration (8)          bands() of the residual of a quadratic fit to each coordinate, x and y averaged
      coverage (1)           share of frames in the +-1 s span with a track point
      size (1)               log1p of the target area A of the window's crop"""
    fk = (k + 0.5) / K * n_frames
    m = np.abs(tr[:, 0] - fk) <= HALF
    if m.sum() < 10: return None
    s = tr[m]; g = np.arange(max(s[0, 0], fk - HALF), min(s[-1, 0], fk + HALF) + 1)
    if len(g) < 10: return None
    x, y, z = np.interp(g, s[:, 0], s[:, 1] / W), np.interp(g, s[:, 0], s[:, 2] / H), np.interp(g, s[:, 0], s[:, 3])
    pol = np.interp(g, s[:, 0], s[:, 4]) if s.shape[1] > 4 else np.zeros_like(z)
    p = np.stack([x, y], 1); v = np.diff(p, axis=0) * fps; a = np.diff(v, axis=0) * fps; j = np.diff(a, axis=0) * fps
    sp = np.linalg.norm(v, axis=1); am = np.linalg.norm(a, axis=1)
    hd = np.unwrap(np.arctan2(v[:, 1], v[:, 0])); dh = np.abs(np.diff(hd)) * (sp[1:] > 1e-3)
    curv = np.abs(v[1:, 0] * a[:, 1] - v[1:, 1] * a[:, 0]) / np.maximum(sp[1:] ** 3, 1e-9)
    path = sp.sum() / fps; net = np.linalg.norm(p[-1] - p[0])
    tt = np.arange(len(g)) / fps; res = np.stack([c - np.polyval(np.polyfit(tt, c, 2), tt) for c in (x, y)], 1)
    kin = [lg(sp.mean()), lg(sp.std()), lg(sp.max()), lg(np.median(sp)), np.sign(v[:, 0].mean()) * lg(v[:, 0].mean()),
           np.sign(v[:, 1].mean()) * lg(v[:, 1].mean()), lg(np.abs(v[:, 0]).mean()), lg(np.abs(v[:, 1]).mean()),
           lg(am.mean()), lg(am.std()), lg(np.linalg.norm(j, axis=1).mean()) if len(j) else 0.0,
           float(dh.mean()) if len(dh) else 0.0, float(np.log(np.median(curv) + 1e-6)) if len(curv) else 0.0,
           net / max(path, 1e-9), lg(res.std(0).mean())]
    za = np.abs(z); pho = [np.log1p(za.mean()), np.log1p(z.std()), z.std() / (za.mean() + 1e-6), float(pol.mean())]
    vib = bands(res[:, 0], fps) + bands(res[:, 1], fps)
    f = np.concatenate([kin, pho, bands(z, fps), vib / 2, [m.sum() / (2 * HALF + 1)], [np.log1p(area)]]).astype(np.float32)
    return np.nan_to_num(f, nan=0.0, posinf=0.0, neginf=0.0)
